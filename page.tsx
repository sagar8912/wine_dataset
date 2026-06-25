"use client"

import { useState } from "react"
import { AppSidebar } from "@/components/global/navigation/app-sidebar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useCurrentUser } from "@/lib/current-user-context"
import { DashboardWelcomeBar } from "@/components/dashboard/dashboard-welcome-bar"
import { DashboardPersonal } from "@/components/dashboard/dashboard-personal"
import { DashboardOrgSummary } from "@/components/dashboard/dashboard-org-summary"

export default function DashboardPage() {
  const { isLoading } = useCurrentUser()
  const [activeTab, setActiveTab] = useState("personal")

  return (
    <div className="flex h-full overflow-hidden">
      <AppSidebar />
      <main className="flex-1 overflow-y-auto bg-gradient-to-b from-background to-muted/20">
        <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
          <DashboardWelcomeBar loading={isLoading} />

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="h-9">
              <TabsTrigger value="personal" className="text-sm">
                Personal
              </TabsTrigger>
              <TabsTrigger value="organization" className="text-sm">
                Organization
              </TabsTrigger>
            </TabsList>

            <TabsContent value="personal" className="mt-6">
              <DashboardPersonal loading={isLoading} />
            </TabsContent>

            <TabsContent value="organization" className="mt-6">
              <DashboardOrgSummary />
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}
