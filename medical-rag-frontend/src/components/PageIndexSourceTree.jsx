import React, { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';

const PageIndexSourceTree = ({ sources = [] }) => {
  const [expanded, setExpanded] = useState({});

  if (!sources || sources.length === 0) {
    return (
      <div className="text-gray-500 text-sm p-4">
        No source pages available
      </div>
    );
  }

  const toggleExpand = (id) => {
    setExpanded((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const renderSection = (section, level = 0, pageIndex) => {
    const sectionId = `${pageIndex}_${section.id}`;
    const hasChildren = section.children && section.children.length > 0;
    const isExpanded = expanded[sectionId];

    return (
      <div key={sectionId} style={{ marginLeft: `${level * 16}px` }}>
        <div className="flex items-start gap-2 py-1.5 hover:bg-gray-900 px-2 rounded cursor-pointer"
             onClick={() => hasChildren && toggleExpand(sectionId)}>
          {hasChildren ? (
            isExpanded ? (
              <ChevronDown size={16} className="text-gray-400 flex-shrink-0 mt-0.5" />
            ) : (
              <ChevronRight size={16} className="text-gray-400 flex-shrink-0 mt-0.5" />
            )
          ) : (
            <div className="w-4" />
          )}
          
          <div className="flex-1 min-w-0">
            <div className="text-gray-300 text-sm font-medium truncate">
              {section.title}
            </div>
            {section.content && level === 0 && (
              <div className="text-gray-500 text-xs mt-1 line-clamp-2">
                {section.content.substring(0, 100)}...
              </div>
            )}
          </div>

          <div className="flex-shrink-0 text-gray-500 text-xs">
            H{section.level}
          </div>
        </div>

        {hasChildren && isExpanded && (
          <div>
            {section.children.map((child) =>
              renderSection(child, level + 1, pageIndex)
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-gray-950 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-gray-800">
        <FileText size={18} className="text-gray-400" />
        <h3 className="text-gray-300 font-semibold">
          Source Pages ({sources.length})
        </h3>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {sources.map((page, idx) => (
          <div key={page.page_id || idx} className="mb-3">
            <div className="flex items-center gap-2 px-2 py-2 bg-gray-900 rounded border-l-2 border-blue-600 mb-2">
              <FileText size={14} className="text-blue-500 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-gray-200 text-sm font-medium truncate">
                  Page {page.page_number || page.page_id}
                </div>
                {page.filename && (
                  <div className="text-gray-500 text-xs">
                    {page.filename}{typeof page.score === "number" ? ` · Relevance: ${page.score.toFixed(3)}` : ""}
                  </div>
                )}
              </div>
            </div>

            {page.sections && page.sections.length > 0 ? (
              <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
                {page.sections.map((section) =>
                  renderSection(section, 0, page.page_id || idx)
                )}
              </div>
            ) : (
              <div className="text-gray-500 text-xs px-4 py-2">
                {page.content ? page.content.substring(0, 150) : 'No content preview'}...
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default PageIndexSourceTree;