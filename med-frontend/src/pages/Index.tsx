
import React, { useState } from 'react';
import { ChatInterface } from '@/components/ChatInterface';
import { Sidebar } from '@/components/Sidebar';
import { Header } from '@/components/Header';

const Index = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Mobile Header */}
      <Header onToggleSidebar={toggleSidebar} />
      
      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          isOpen={sidebarOpen}
          onToggle={toggleSidebar}
          selectedConversation={selectedConversation}
          onSelectConversation={setSelectedConversation}
        />
        
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          <ChatInterface selectedConversation={selectedConversation} />
        </div>
      </div>
    </div>
  );
};

export default Index;
