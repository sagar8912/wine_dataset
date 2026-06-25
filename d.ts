import { type NextRequest, NextResponse } from "next/server"
import { createServiceClient } from "@/lib/supabase/service"
import { requireAuthOrFail, isAdmin } from "@/lib/auth-helpers"
import { NotificationService } from "@/lib/services/notification-service"
import { NOTIFICATION_TYPES } from "@/lib/constants/notification-types"
import { getRequestIdFromRequest, logErrorWithRequestId, generateRequestId } from "@theoakbridgeway/utils/api-helpers"
import type { SupabaseTeamMemberRow, SupabaseUserProfileRow, SupabaseQueryResponse, SupabaseSingleResponse } from "@theoakbridgeway/types"
import { TRANSFERABLE_ITEM_STATUSES } from "@/lib/items/status-transitions"

// Type for team member with user profile relation
type TeamMemberWithUser = Omit<SupabaseTeamMemberRow, "joined_at" | "user" | "team" | "id"> & {
  id: string | null
  joined_at: string | null
  user: Pick<SupabaseUserProfileRow, "id" | "full_name" | "email" | "avatar_url" | "system_role"> | null
}

// GET /api/team-members?userId=xxx OR ?teamId=xxx
export async function GET(request: NextRequest) {
  try {
    // Always require authentication first (security: prevent cookie-based bypass)
    const authResult = await requireAuthOrFail(request)
    if (authResult instanceof NextResponse) {
      return authResult // Return 401 error
    }
    
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("userId")
    const teamId = searchParams.get("teamId")

    // Task 7.1: team_members SELECT RLS relaxed to all authenticated users — session client
    // now works for non-members too. Service client only needed for impersonation.
    const supabase = authResult.isImpersonating ? createServiceClient() : authResult.supabase

    if (!userId && !teamId) {
      return NextResponse.json({ error: "userId or teamId is required" }, { status: 400 })
    }

    // Try with foreign key join first - use simpler syntax that doesn't require constraint name
    let query = supabase
      .from("team_members")
      .select(
        `
        *,
        user:user_profiles(id, full_name, email, avatar_url, system_role)
      `
      )

    if (userId) {
      query = query.eq("user_id", userId)
    }
    if (teamId) {
      query = query.eq("team_id", teamId)
    }

    const { data: teamMembersRaw, error: queryError } = await query.order("joined_at", { ascending: false })
    let error = queryError
    
    // Convert user array to single object for consistent typing (Supabase joins return arrays)
    let teamMembers: TeamMemberWithUser[] | null = teamMembersRaw?.map((m: any) => ({
      ...m,
      user: Array.isArray(m.user) ? m.user[0] || null : m.user || null,
    })) || null

    // If the join fails or returns no data, try without the join and fetch users separately
    if (error || !teamMembers || teamMembers.length === 0) {
      // Fallback: query without join
      let fallbackQuery = supabase.from("team_members").select("*")

      if (userId) {
        fallbackQuery = fallbackQuery.eq("user_id", userId)
      }
      if (teamId) {
        fallbackQuery = fallbackQuery.eq("team_id", teamId)
      }

      const { data: teamMembersFallback, error: fallbackError }: SupabaseQueryResponse<SupabaseTeamMemberRow> = await fallbackQuery.order("joined_at", { ascending: false })

      if (!fallbackError && teamMembersFallback && teamMembersFallback.length > 0) {
        error = null

        // Fetch user data for all members
        const userIds = [...new Set(teamMembersFallback.map((m: SupabaseTeamMemberRow) => m.user_id).filter(Boolean))]
        if (userIds.length > 0) {
          const { data: userProfiles, error: usersError }: SupabaseQueryResponse<Pick<SupabaseUserProfileRow, "id" | "full_name" | "email" | "avatar_url" | "system_role">> = await supabase
            .from("user_profiles")
            .select("id, full_name, email, avatar_url, system_role")
            .in("id", userIds)

          if (!usersError && userProfiles) {
            const usersMap = new Map(userProfiles.map((u: Pick<SupabaseUserProfileRow, "id" | "full_name" | "email" | "avatar_url" | "system_role">) => [u.id, u]))
            teamMembers = teamMembersFallback.map((m: SupabaseTeamMemberRow): TeamMemberWithUser => ({
              ...m,
              user: usersMap.get(m.user_id) || null,
            }))
          } else if (usersError) {
            console.error("[Team Members API GET] Error fetching user profiles in fallback:", usersError)
          }
        }
      } else if (fallbackError) {
        console.error("[Team Members API GET] Fallback query also failed:", fallbackError)
      }
    }

    if (error) {
      console.error("[Team Members API GET] Error fetching team members:", {
        userId,
        teamId,
        error: error.message,
        code: error.code,
        details: error.details,
        hint: error.hint,
      })
      throw error
    }

    // Phase 2a trigger guarantees that teams.owner_id and teams.team_admin_id always
    // have corresponding team_members rows. No synthetic owner/admin injection needed.
    return NextResponse.json({ data: teamMembers || [] })
  } catch (error: unknown) {
    let requestId: string
    try {
      requestId = await getRequestIdFromRequest(request)
    } catch {
      requestId = generateRequestId()
    }
    logErrorWithRequestId(requestId, error, "Error fetching team members")
    const errorMessage = error instanceof Error ? error.message : String(error) || "Internal server error"
    return NextResponse.json({ error: errorMessage }, { status: 500 })
  }
}

// POST /api/team-members - Add user to team
export async function POST(request: NextRequest) {
  const requestId = await getRequestIdFromRequest(request)
  
  try {
    // Require authentication to know who is making the change
    const authResult = await requireAuthOrFail(request)
    if (authResult instanceof NextResponse) {
      console.error("[Team Members API POST] Authentication failed", {
        requestId,
        status: authResult.status,
        statusText: authResult.statusText,
      })
      return authResult
    }
    const { user } = authResult

    const serviceClient = createServiceClient()
    const body = await request.json()
    const { teamId, userId, role } = body

    if (!teamId || !userId) {
      console.error("[Team Members API POST] Missing required fields", {
        requestId,
        hasTeamId: !!teamId,
        hasUserId: !!userId,
      })
      return NextResponse.json({ error: "teamId and userId are required" }, { status: 400 })
    }

    // Fetch team name for notifications (before the RPC call)
    const { data: team } = await serviceClient
      .from("teams")
      .select("id, name")
      .eq("id", teamId)
      .single()

    const { data: rpcResult, error } = await serviceClient.rpc("add_team_member", {
      p_team_id:  teamId,
      p_user_id:  userId,
      p_role:     role || "Team Member",
      p_added_by: user.id,
    })

    if (error) {
      const errorText = error.message?.toLowerCase() ?? ""
      if (error.code === "42501" || errorText.includes("permission_denied")) {
        return NextResponse.json({ error: "Not authorized to manage team membership" }, { status: 403 })
      }
      logErrorWithRequestId(requestId, error, "Database error managing team membership")
      console.error("[Team Members API POST] RPC error:", {
        requestId, teamId, userId, error: error.message, code: error.code,
      })
      throw error
    }

    if (!rpcResult) {
      return NextResponse.json({ error: "Failed to manage team membership - no data returned" }, { status: 500 })
    }

    const data = rpcResult as Record<string, unknown>
    const wasUpdate = data._was_update as boolean
    const previousRole = data._previous_role as string | undefined
    const teamRole = data.role as string

    // Notifications (fire-and-forget)
    if (userId !== user.id) {
      try {
        const { data: actorProfile } = await serviceClient
          .from("user_profiles")
          .select("full_name")
          .eq("id", user.id)
          .single()
        const actorName = actorProfile?.full_name || "Someone"

        if (wasUpdate && previousRole !== teamRole) {
          await NotificationService.createNotification(
            {
              type: NOTIFICATION_TYPES.TEAM_ROLE_CHANGED,
              title: "Team Role Changed",
              message: `${actorName} changed your role in "${team?.name || "the team"}" from ${previousRole} to ${teamRole}.`,
              recipients: [userId],
              teamId,
              metadata: { previousRole, newRole: teamRole, changedBy: user.id },
              excludeUserId: user.id,
            },
            serviceClient
          )
        } else if (!wasUpdate) {
          await NotificationService.createNotification(
            {
              type: NOTIFICATION_TYPES.TEAM_MEMBER_ADDED,
              title: "Added to Team",
              message: `${actorName} added you to "${team?.name || "a team"}" as ${teamRole}.`,
              recipients: [userId],
              teamId,
              metadata: { role: teamRole, addedBy: user.id },
              excludeUserId: user.id,
            },
            serviceClient
          )
        }
      } catch (notifError) {
        logErrorWithRequestId(requestId, notifError as Error, "Error creating notification for team membership change")
      }
    }

    // Strip internal metadata fields before returning to client
    const { _was_update: _w, _previous_role: _p, ...memberData } = data
    void _w; void _p
    return NextResponse.json({ data: memberData })
  } catch (error: unknown) {
    logErrorWithRequestId(requestId, error, "Error in team members POST handler")
    const errorMessage = error instanceof Error ? error.message : "Internal server error"
    return NextResponse.json({ error: errorMessage }, { status: 500 })
  }
}

// DELETE /api/team-members?teamId=xxx&userId=xxx
export async function DELETE(request: NextRequest) {
  const requestId = await getRequestIdFromRequest(request)
  try {
    // Require authentication to know who is making the change
    const authResult = await requireAuthOrFail(request)
    if (authResult instanceof NextResponse) {
      return authResult
    }
    const { user } = authResult

    const supabase = createServiceClient()
    const { searchParams } = new URL(request.url)
    const teamId = searchParams.get("teamId")
    const userId = searchParams.get("userId")

    if (!teamId || !userId) {
      return NextResponse.json({ error: "teamId and userId are required" }, { status: 400 })
    }

    // Role check: only Team Admin/Owner or system Admin+ may remove a member, OR the member
    // may remove themselves (self-leave).
    const serviceForDeleteCheck = createServiceClient()
    const isSelfRemove = user.id === userId
    if (!isSelfRemove) {
      const callerIsAdmin = await isAdmin(serviceForDeleteCheck, user.id)
      if (!callerIsAdmin) {
        const { data: callerMembership } = await serviceForDeleteCheck
          .from("team_members")
          .select("role")
          .eq("team_id", teamId)
          .eq("user_id", user.id)
          .maybeSingle()
        const callerRole = callerMembership?.role ?? ""
        if (callerRole !== "Team Owner" && callerRole !== "Team Admin") {
          return NextResponse.json({ error: "Not authorized to remove team members" }, { status: 403 })
        }
      }
    }

    // Fetch team name for notification before deletion
    const { data: team } = await supabase
      .from("teams")
      .select("id, name")
      .eq("id", teamId)
      .single()

    // Task 7.5: Pre-delete ownership check (Invariant 21).
    // If the member owns active items or non-archived goals on this team, return
    // OWNERSHIP_TRANSFER_REQUIRED (409) instead of proceeding. The caller is responsible
    // for transferring ownership of each record before re-submitting the deletion.
    //
    // KNOWN LIMITATION (TOCTOU): The check and the deletion are not wrapped in a
    // database transaction. A concurrent request could assign a new record to the member
    // after this check passes but before the DELETE executes. The correct long-term fix
    // is to move this guard into a SECURITY DEFINER RPC that performs the check and delete
    // atomically. Tracked as a follow-up (Task 7.6 scope or new task).
    const [itemsResult, goalsResult] = await Promise.all([
      supabase
        .from("items")
        .select("id, title, status, item_type")
        .eq("team_id", teamId)
        .eq("owner_id", userId)
        .in("status", TRANSFERABLE_ITEM_STATUSES),
      supabase
        .from("goals")
        .select("id, title, status")
        .eq("team_id", teamId)
        .eq("owner_id", userId)
        .is("archived_at", null),
    ])

    const ownedItems = itemsResult.data ?? []
    const ownedGoals = goalsResult.data ?? []

    if (ownedItems.length > 0 || ownedGoals.length > 0) {
      // HTTP 409 Conflict: a business precondition prevents deletion.
      // The client must transfer ownership of all listed records before re-submitting.
      return NextResponse.json(
        {
          status: "OWNERSHIP_TRANSFER_REQUIRED",
          items: ownedItems,
          goals: ownedGoals,
        },
        { status: 409 },
      )
    }

    const { error } = await supabase
      .from("team_members")
      .delete()
      .eq("team_id", teamId)
      .eq("user_id", userId)

    if (error) throw error

    // Create notification for user being removed from team (only if user is not the one making the change)
    if (userId !== user.id) {
      try {
        const { data: removerProfile } = await supabase
          .from("user_profiles")
          .select("full_name")
          .eq("id", user.id)
          .single()

        const removerName = removerProfile?.full_name || "Someone"

        await NotificationService.createNotification(
          {
            type: NOTIFICATION_TYPES.TEAM_MEMBER_REMOVED,
            title: "Removed from Team",
            message: `${removerName} removed you from "${team?.name || "a team"}".`,
            recipients: [userId],
            teamId,
            metadata: {
              removedBy: user.id,
            },
            excludeUserId: user.id,
          },
          supabase
        )
      } catch (notifError) {
        logErrorWithRequestId(requestId, notifError as Error, "Error creating notification for team member removal")
      }
    }

    return NextResponse.json({ success: true })
  } catch (error: unknown) {
    logErrorWithRequestId(requestId, error, "Error deleting team member")
    const errorMessage = error instanceof Error ? error.message : "Internal server error"
    return NextResponse.json({ error: errorMessage }, { status: 500 })
  }
}
