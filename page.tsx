"use client"

import { SidebarShell } from "@/components/global/layout/sidebar-shell"
import { AppSidebar } from "@/components/global/navigation/app-sidebar"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { SearchableSelect } from "@/components/ui/searchable-select"
import { Plus, Shield, Users, Key } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { useToast } from "@/hooks/use-toast"
import { DataService } from "@/lib/data-service"
import type { Team } from "@theoakbridgeway/types"
import { fetchWithAuth } from "@/lib/fetch-with-auth"
import { useCurrentUser } from "@/lib/current-user-context"
import { NewUserDialog } from "@/components/users"

export interface AdminUserRow {
  userId: string
  userName: string
  email: string
  systemRole: string
  teamRoles: any[]
  department: string
  title: string
  lastLogin: string
  primaryTeamId?: string | null
  primaryTeamName?: string | null
}

export interface AdminUserFilters {
  searchQuery: string
  role: string
  primaryTeamId: string
}

export function filterAdminUsers(users: AdminUserRow[], filters: AdminUserFilters): AdminUserRow[] {
  const normalizedSearch = filters.searchQuery.trim().toLowerCase()
  const normalizedRole = filters.role.toLowerCase()
  const primaryTeamFilter = filters.primaryTeamId

  return users.filter((user) => {
    const departmentValue = (user.department || "Unknown").trim()
    const matchesSearch =
      normalizedSearch.length === 0 ||
      user.userName.toLowerCase().includes(normalizedSearch) ||
      user.email.toLowerCase().includes(normalizedSearch) ||
      departmentValue.toLowerCase().includes(normalizedSearch) ||
      (user.primaryTeamName || "").toLowerCase().includes(normalizedSearch)

    if (!matchesSearch) {
      return false
    }

    const matchesRole =
      normalizedRole === "all" || user.systemRole.toLowerCase() === normalizedRole

    if (!matchesRole) {
      return false
    }

    const matchesPrimaryTeam =
      primaryTeamFilter === "all" ||
      (primaryTeamFilter === "none" ? !user.primaryTeamId : user.primaryTeamId === primaryTeamFilter)

    return matchesPrimaryTeam
  })
}

export function countUsersByRole(users: AdminUserRow[]): Record<string, number> {
  return users.reduce((acc, user) => {
    const roleKey = user.systemRole || "Unknown"
    acc[roleKey] = (acc[roleKey] || 0) + 1
    return acc
  }, {} as Record<string, number>)
}

const SYSTEM_ROLES = ["Super Admin", "Admin", "Manager", "User", "Viewer", "No Role Assigned"] as const

export default function AdminUsersPage() {
  const { currentUser } = useCurrentUser()
  const [users, setUsers] = useState<AdminUserRow[]>([])
  const [teams, setTeams] = useState<Team[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [editingUser, setEditingUser] = useState<(AdminUserRow & { primaryTeamId?: string }) | null>(null)
  const [saving, setSaving] = useState(false)
  const [resettingPasswordFor, setResettingPasswordFor] = useState<string | null>(null)
  const [filters, setFilters] = useState<AdminUserFilters>({
    searchQuery: "",
    role: "all",
    primaryTeamId: "all",
  })
  const { toast } = useToast()
  
  // Check if current user is Admin or Super Admin
  const isAdmin = currentUser.systemRole === "Admin" || currentUser.systemRole === "Super Admin"

  useEffect(() => {
    loadUsers()
    loadTeams()
  }, [])

  const loadTeams = async () => {
    try {
      const fetchedTeams = await DataService.getTeams()
      setTeams(fetchedTeams || [])
    } catch (error) {
      console.error("[AdminUsers] Error loading teams:", error)
      // Set empty array on error to prevent hanging
      setTeams([])
    }
  }

  const loadUsers = async () => {
    try {
      setLoading(true)
      const [profilesResponse, userRolesResponse] = await Promise.all([
        fetch("/api/profiles"),
        fetch("/api/user-roles"),
      ])

      // Parse responses - handle errors properly
      let profilesData, userRolesData
      
      try {
        profilesData = await profilesResponse.json()
      } catch (error) {
        const errorText = await profilesResponse.text().catch(() => "Could not read error")
        throw new Error(`Profiles API error (${profilesResponse.status}): ${errorText}`)
      }

      try {
        userRolesData = await userRolesResponse.json()
      } catch (error) {
        const errorText = await userRolesResponse.text().catch(() => "Could not read error")
        throw new Error(`User Roles API error (${userRolesResponse.status}): ${errorText}`)
      }

      // Check for API-level errors in response
      if (!profilesResponse.ok || profilesData?.error) {
        throw new Error(`Profiles API error (${profilesResponse.status}): ${profilesData?.error || "Unknown error"}`)
      }

      // user-roles endpoint is optional - continue even if it fails or returns no data
      const profiles = profilesData?.data || []
      
      let userRolesDataArray = []
      if (userRolesResponse.ok && !userRolesData?.error) {
        userRolesDataArray = userRolesData?.data || []
      } else {
        // User roles API failed or returned error, continuing without roles
      }

      // Create a map of user_id to system_role for quick lookup
      const userRolesMap = new Map()
      userRolesDataArray?.forEach((ur: any) => {
        userRolesMap.set(ur.userId, ur.systemRole)
      })

      // Combine profiles with their system roles from user_profiles table
      // (user_roles table has been removed)
      const formattedUsers: AdminUserRow[] = (profiles || []).map((profile: any) => ({
        userId: profile.id,
        userName: profile.name || "Unknown",
        email: profile.email || "",
        systemRole: userRolesMap.get(profile.id) || profile.role || profile.system_role || "User",
        teamRoles: [],
        department: (profile.department || "Unknown").trim() || "Unknown",
        title: profile.title || "",
        lastLogin: profile.updated_at || new Date().toISOString(),
      }))

      const primaryTeamResults = await Promise.all(
        formattedUsers.map(async (user) => {
          try {
            const response = await fetchWithAuth(`/api/team-members?userId=${user.userId}`)
            if (!response.ok) {
              return { userId: user.userId, teamId: null as string | null }
            }
            const payload = await response.json()
            const memberships = Array.isArray(payload?.data) ? payload.data : []
            if (memberships.length > 0) {
              const primaryMembership = memberships[0]
              return { userId: user.userId, teamId: primaryMembership.team_id as string | null }
            }
            return { userId: user.userId, teamId: null as string | null }
          } catch (primaryTeamError) {
            console.error("[AdminUsers] Error fetching primary team for user:", user.userId, primaryTeamError)
            return { userId: user.userId, teamId: null as string | null }
          }
        }),
      )

      const primaryTeamMap = new Map(primaryTeamResults.map((entry) => [entry.userId, entry.teamId]))

      const usersWithPrimaryTeam = formattedUsers.map((user) => ({
        ...user,
        primaryTeamId: primaryTeamMap.get(user.userId) ?? null,
      }))

      setUsers(usersWithPrimaryTeam)
    } catch (error: any) {
      console.error("[AdminUsers] Error loading users:", error instanceof Error ? error.message : error)
      const errorMessage = error?.message || error?.toString() || "Unknown error"
      
      // Set empty array on error to prevent hanging
      setUsers([])
      toast({
        title: "Error",
        description: `Failed to load users: ${errorMessage}`,
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const handleEditUser = async () => {
    if (!editingUser || !editingUser.userName || !editingUser.email) {
      toast({
        title: "Validation Error",
        description: "Name and email are required",
        variant: "destructive",
      })
      return
    }

    try {
      setSaving(true)
      const response = await fetch("/api/profiles", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: editingUser.userId,
          name: editingUser.userName,
          email: editingUser.email,
          department: editingUser.department || "",
          title: editingUser.title || "",
          primaryTeamId: editingUser.primaryTeamId || undefined,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const errorMessage = errorData?.error || `HTTP ${response.status}: Failed to update user`
        throw new Error(errorMessage)
      }

      if (editingUser.systemRole && editingUser.systemRole !== "No Role Assigned") {
        const roleResponse = await fetch("/api/user-roles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId: editingUser.userId,
            systemRole: editingUser.systemRole,
          }),
        })

        if (!roleResponse.ok) {
          const err = await roleResponse.json().catch(() => ({} as any))
          const msg = err?.error || `HTTP ${roleResponse.status}: Failed to update user role`
          console.error("[AdminUsers] Failed to update user role", msg)
          throw new Error(msg)
        }
      }

      toast({
        title: "Success",
        description: "User updated successfully",
      })

      setShowEditDialog(false)
      setEditingUser(null)

      await loadUsers()
    } catch (error: any) {
      console.error("[AdminUsers] Error updating user:", error?.message || error)
      toast({
        title: "Error",
        description: error?.message || "Failed to update user",
        variant: "destructive",
      })
    } finally {
      setSaving(false)
    }
  }

  const handleResetPassword = async (userId: string, userName: string, userEmail: string) => {
    try {
      setResettingPasswordFor(userId)
      
      const response = await fetchWithAuth("/api/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Failed to reset password" }))
        throw new Error(errorData.error || "Failed to reset password")
      }

      const { data } = await response.json()
      
      // Show appropriate message based on whether email was sent
      if (data.emailSent) {
        toast({
          title: "Password Reset Email Sent",
          description: `A password reset email has been sent to ${userEmail}.`,
          duration: 10000,
        })
      } else {
        toast({
          title: "Password Reset Link Generated",
          description: `Reset link for ${userName} (${userEmail}) has been generated. Check console for the link.`,
          duration: 10000,
        })
      }
      
    } catch (error: any) {
      console.error("[AdminUsers] Error resetting password:", error)
      toast({
        title: "Error",
        description: error?.message || "Failed to reset password",
        variant: "destructive",
      })
    } finally {
      setResettingPasswordFor(null)
    }
  }

  const getSystemRoleColor = (role: string) => {
    switch (role) {
      case "Super Admin":
        return "bg-purple-500"
      case "Admin":
        return "bg-blue-500"
      case "Manager":
        return "bg-green-500"
      case "User":
        return "bg-gray-500"
      case "Viewer":
        return "bg-slate-500"
      case "No Role Assigned":
        return "bg-orange-500"
      default:
        return "bg-gray-500"
    }
  }

  const teamNameMap = useMemo(() => {
    const entries = teams.map((team) => [team.id, team.name] as const)
    return new Map(entries)
  }, [teams])

  const usersWithPrimaryTeamNames = useMemo(
    () =>
      users.map((user) => {
        const primaryTeamName =
          user.primaryTeamId && teamNameMap.size > 0
            ? teamNameMap.get(user.primaryTeamId) || "Unknown Team"
            : user.primaryTeamId
            ? "Unknown Team"
            : "No Primary Team"
        return {
          ...user,
          primaryTeamName,
        }
      }),
    [users, teamNameMap],
  )

  const filteredUsers = useMemo(
    () => filterAdminUsers(usersWithPrimaryTeamNames, filters),
    [usersWithPrimaryTeamNames, filters],
  )

  const overallRoleCounts = useMemo(
    () => countUsersByRole(usersWithPrimaryTeamNames),
    [usersWithPrimaryTeamNames],
  )

  const primaryTeamOptions = useMemo(() => {
    const optionMap = new Map<string, string>()
    teams.forEach((team) => optionMap.set(team.id, team.name))
    usersWithPrimaryTeamNames.forEach((user) => {
      if (user.primaryTeamId && !optionMap.has(user.primaryTeamId)) {
        optionMap.set(user.primaryTeamId, user.primaryTeamName || "Unknown Team")
      }
    })
    return Array.from(optionMap.entries()).sort((a, b) => a[1].localeCompare(b[1]))
  }, [teams, usersWithPrimaryTeamNames])

  const clearFilters = () => {
    setFilters({
      searchQuery: "",
      role: "all",
      primaryTeamId: "all",
    })
  }

  if (loading) {
    return (
      <SidebarShell contentClassName="p-8">
        <div className="mx-auto max-w-7xl">
          <p className="text-muted-foreground">Loading users...</p>
        </div>
      </SidebarShell>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      <AppSidebar />
      <main className="flex-1 overflow-y-auto p-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-8 flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold text-foreground">User Management</h1>
              <p className="mt-2 text-muted-foreground">Manage user roles and permissions across the organization</p>
            </div>
            <Button onClick={() => setShowAddDialog(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add User
            </Button>
          </div>

          <div className="mb-6 grid gap-4 md:grid-cols-5">
            <Card className="p-4">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-purple-500" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {overallRoleCounts["Super Admin"] ?? 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Super Admins</div>
                </div>
              </div>
            </Card>
            <Card className="p-4">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-blue-500" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {overallRoleCounts["Admin"] ?? 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Admins</div>
                </div>
              </div>
            </Card>
            <Card className="p-4">
              <div className="flex items-center gap-2">
                <Users className="h-5 w-5 text-green-500" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {overallRoleCounts["Manager"] ?? 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Managers</div>
                </div>
              </div>
            </Card>
            <Card className="p-4">
              <div className="flex items-center gap-2">
                <Users className="h-5 w-5 text-gray-500" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {overallRoleCounts["User"] ?? 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Users</div>
                </div>
              </div>
            </Card>
            <Card className="p-4">
              <div className="flex items-center gap-2">
                <Users className="h-5 w-5 text-slate-500" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {overallRoleCounts["Viewer"] ?? 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Viewers</div>
                </div>
              </div>
            </Card>
          </div>

          {overallRoleCounts["No Role Assigned"] ? (
            <Card className="mb-6 border-orange-200 bg-orange-50 p-4">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-orange-500" />
                <div>
                  <div className="text-sm font-medium text-orange-900">
                    {overallRoleCounts["No Role Assigned"]} user(s) without assigned roles
                  </div>
                  <div className="text-xs text-orange-700">
                    These users need to be assigned a system role to access the application
                  </div>
                </div>
              </div>
            </Card>
          ) : null}

          <Card className="mb-6 p-4">
            <div className="grid gap-4 md:grid-cols-4">
              <div className="md:col-span-2">
                <Label htmlFor="user-search">Search Users</Label>
                <Input
                  id="user-search"
                  value={filters.searchQuery}
                  onChange={(event) =>
                    setFilters((prev) => ({ ...prev, searchQuery: event.target.value }))
                  }
                  placeholder="Search by name, email, or department"
                  className="mt-2"
                />
              </div>
              <div>
                <Label htmlFor="role-filter">System Role</Label>
                <Select
                  value={filters.role}
                  onValueChange={(value) => setFilters((prev) => ({ ...prev, role: value }))}
                >
                  <SelectTrigger id="role-filter" className="mt-2">
                    <SelectValue placeholder="All roles" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All roles</SelectItem>
                    {SYSTEM_ROLES.map((role) => (
                      <SelectItem key={role} value={role}>
                        {role}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="primary-team-filter">Primary Team</Label>
                <div className="mt-2">
                  <SearchableSelect
                    value={filters.primaryTeamId}
                    onValueChange={(val) => setFilters((prev) => ({ ...prev, primaryTeamId: val ?? "all" }))}
                    placeholder="Filter by primary team"
                    options={[
                      { value: "all", label: "All primary teams" },
                      { value: "none", label: "No Primary Team" },
                      ...primaryTeamOptions.map(([teamId, teamName]) => ({
                        value: teamId,
                        label: teamName,
                      })),
                    ]}
                    variant="inline"
                    listClassName="max-h-64 overflow-y-auto"
                  />
                </div>
              </div>
            </div>
            {(filters.searchQuery || filters.role !== "all" || filters.primaryTeamId !== "all") && (
              <div className="mt-4 flex justify-end">
                <Button variant="outline" size="sm" onClick={clearFilters}>
                  Clear filters
                </Button>
              </div>
            )}
          </Card>

          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead>Email</TableHead>
                  <TableHead>System Role</TableHead>
                  <TableHead>Team Roles</TableHead>
                  <TableHead>Primary Team</TableHead>
                  <TableHead>Last Login</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredUsers.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="py-6 text-center text-sm text-muted-foreground">
                      No users match the current filters.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredUsers.map((user) => (
                    <TableRow key={user.userId}>
                      <TableCell className="font-medium text-foreground">{user.userName}</TableCell>
                      <TableCell className="text-muted-foreground">{user.email}</TableCell>
                      <TableCell>
                        <Badge className={getSystemRoleColor(user.systemRole)}>{user.systemRole}</Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {user.teamRoles.length > 0 ? (
                            user.teamRoles.map((teamRole: any) => (
                              <Badge key={teamRole.teamId} variant="outline" className="text-xs">
                                {teamRole.role}
                              </Badge>
                            ))
                          ) : (
                            <span className="text-xs text-muted-foreground">None</span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {user.primaryTeamName || "No Primary Team"}
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {new Date(user.lastLogin).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={async () => {
                              // Fetch user's primary team (first team_members record)
                              let primaryTeamId = ""
                              try {
                                const teamMembersResponse = await fetchWithAuth(`/api/team-members?userId=${user.userId}`)
                                if (teamMembersResponse.ok) {
                                  const { data: teamMembers } = await teamMembersResponse.json()
                                  if (teamMembers && teamMembers.length > 0) {
                                    primaryTeamId = teamMembers[0].team_id
                                  }
                                }
                              } catch (error) {
                                console.error("[AdminUsers] Error fetching primary team:", error)
                              }

                              setEditingUser({
                                userId: user.userId,
                                userName: user.userName || "",
                                email: user.email || "",
                                systemRole: user.systemRole === "No Role Assigned" ? "User" : user.systemRole,
                                department: user.department || "",
                                title: user.title || "",
                                primaryTeamId: primaryTeamId || "",
                                teamRoles: user.teamRoles || [],
                                lastLogin: user.lastLogin,
                                primaryTeamName: user.primaryTeamName || null,
                              })
                              setShowEditDialog(true)
                            }}
                          >
                            Edit
                          </Button>
                          {isAdmin && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleResetPassword(user.userId, user.userName, user.email)}
                              disabled={resettingPasswordFor === user.userId}
                              title="Reset Password"
                            >
                              <Key className="h-4 w-4" />
                              {resettingPasswordFor === user.userId ? "..." : ""}
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </Card>

          <div className="mt-8">
            <h2 className="mb-4 text-2xl font-semibold text-foreground">Role Descriptions</h2>
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="p-4">
                <h3 className="mb-2 font-semibold text-foreground">System Roles</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>
                    <strong className="text-foreground">Super Admin:</strong> Full system access, can manage all users
                    and settings
                  </li>
                  <li>
                    <strong className="text-foreground">Admin:</strong> Can manage users within their department
                  </li>
                  <li>
                    <strong className="text-foreground">Manager:</strong> Can manage team members and view reports
                  </li>
                  <li>
                    <strong className="text-foreground">User:</strong> Standard access to create and manage own items
                  </li>
                  <li>
                    <strong className="text-foreground">Viewer:</strong> Read-only access to assigned teams
                  </li>
                  <li>
                    <strong className="text-foreground">No Role Assigned:</strong> Users without any system role
                    assigned
                  </li>
                </ul>
              </Card>
              <Card className="p-4">
                <h3 className="mb-2 font-semibold text-foreground">Team Roles</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>
                    <strong className="text-foreground">Team Owner:</strong> Full control over team list and rocks
                  </li>
                  <li>
                    <strong className="text-foreground">Team Admin:</strong> Can manage team items and members
                  </li>
                  <li>
                    <strong className="text-foreground">Team Member:</strong> Can create and edit own items
                  </li>
                  <li>
                    <strong className="text-foreground">Team Viewer:</strong> Read-only access to team list
                  </li>
                </ul>
              </Card>
            </div>
          </div>
        </div>
      </main>

      <NewUserDialog
        open={showAddDialog}
        onOpenChange={setShowAddDialog}
        teams={teams}
        onUserCreated={loadUsers}
      />

      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit User</DialogTitle>
            <DialogDescription>Update user information and role</DialogDescription>
          </DialogHeader>
          {editingUser && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="edit-name">Name *</Label>
                <Input
                  id="edit-name"
                  value={editingUser.userName}
                  onChange={(e) => setEditingUser({ ...editingUser, userName: e.target.value })}
                  placeholder="John Doe"
                />
              </div>
              <div>
                <Label htmlFor="edit-email">Email *</Label>
                <Input
                  id="edit-email"
                  type="email"
                  value={editingUser.email}
                  onChange={(e) => setEditingUser({ ...editingUser, email: e.target.value })}
                  placeholder="john.doe@company.com"
                />
              </div>
              <div>
                <Label htmlFor="edit-role">System Role</Label>
                <Select
                  value={editingUser.systemRole}
                  onValueChange={(value) => setEditingUser({ ...editingUser, systemRole: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Super Admin">Super Admin</SelectItem>
                    <SelectItem value="Admin">Admin</SelectItem>
                    <SelectItem value="Manager">Manager</SelectItem>
                    <SelectItem value="User">User</SelectItem>
                    <SelectItem value="Viewer">Viewer</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="edit-department">Department</Label>
                <Input
                  id="edit-department"
                  value={editingUser.department}
                  onChange={(e) => setEditingUser({ ...editingUser, department: e.target.value })}
                  placeholder="Engineering"
                />
              </div>
              <div>
                <Label htmlFor="edit-title">Job Title</Label>
                <Input
                  id="edit-title"
                  value={editingUser.title}
                  onChange={(e) => setEditingUser({ ...editingUser, title: e.target.value })}
                  placeholder="Senior Engineer"
                />
              </div>
              <div>
                <Label htmlFor="edit-primaryTeam">Primary Team</Label>
                <SearchableSelect
                  value={editingUser.primaryTeamId || "none"}
                  onValueChange={(val) => {
                    const normalized = val && val !== "none" ? val : ""
                    setEditingUser({ ...editingUser, primaryTeamId: normalized })
                  }}
                  placeholder="Select primary team (optional)"
                  options={[
                    { value: "none", label: "No Primary Team" },
                    ...teams.map((team) => ({
                      value: team.id,
                      label: team.name,
                      description: team.department || undefined,
                      meta: team.hierarchyLevel,
                    })),
                  ]}
                  variant="inline"
                  listClassName="max-h-64 overflow-y-auto"
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditDialog(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={handleEditUser} disabled={saving}>
              {saving ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

