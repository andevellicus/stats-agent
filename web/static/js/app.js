// Auto-scroll to bottom when new messages are added
function observeMessages() {
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        const observer = new MutationObserver(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        });
        observer.observe(messagesContainer, { childList: true });
    }
}

// Focus input on page load
function focusInput() {
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.focus();
    }
}

// Toggle code block visibility
function toggleCodeBlock(element) {
    const content = element.nextElementSibling;
    const icon = element.querySelector('svg');
    content.classList.toggle('hidden');
    icon.classList.toggle('rotate-180');
}

// Handle form submission on Enter key
function submitOnEnter(event) {
    if (event.keyCode == 13 && !event.shiftKey) {
        event.preventDefault();
        event.target.form.requestSubmit();
    }
}

// Initialize all scripts on page load
document.addEventListener('DOMContentLoaded', () => {
    observeMessages();
    focusInput();
    initiateSSE();
});

// Re-initialize after HTMX swaps content
document.body.addEventListener('htmx:afterSwap', function(event) {
    focusInput();
    initiateSSE();
});

function initiateSSE() {
    const loaders = document.querySelectorAll('.sse-loader:not([data-sse-initialized])');
    loaders.forEach(function(loader) {
        loader.setAttribute('data-sse-initialized', 'true');
        const sessionId = loader.getAttribute('data-session-id');
        const messageId = loader.getAttribute('data-message-id');

        if (!sessionId || !messageId) return;

        const eventSource = new EventSource('/chat/stream?session_id=' + encodeURIComponent(sessionId) + '&user_message_id=' + encodeURIComponent(messageId));
        
        let contentBuffer = '';
        let agentMessageContainer = null;

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'connection_established':
                    console.log('SSE connection established.');
                    break;
                case 'remove_loader':
                    const loadingIndicator = document.getElementById(data.content);
                    if (loadingIndicator) {
                        loadingIndicator.remove();
                    }
                    break;
                case 'create_container':
                    const agentMessageId = 'agent-msg-' + data.content;
                    agentMessageContainer = document.createElement('div');
                    agentMessageContainer.id = agentMessageId;
                    agentMessageContainer.innerHTML = `
                        <div class="flex justify-start">
                            <div class="agent-output">
                                <div class="font-semibold text-sm text-gray-600">Stats Agent</div>
                                <div id="content-${agentMessageId}"></div>
                            </div>
                        </div>
                    `;
                    document.getElementById('messages').appendChild(agentMessageContainer);
                    break;
                case 'chunk':
                    if (agentMessageContainer) {
                        const contentDiv = document.getElementById('content-' + agentMessageContainer.id);
                        contentBuffer += data.content;

                        // Show raw content in real-time (no processing yet)
                        contentDiv.innerHTML = '<pre class="font-mono text-sm whitespace-pre-wrap">' + contentBuffer + '</pre>';
                    }
                    break;
                case 'final_content':
                    if (agentMessageContainer) {
                        const contentDiv = document.getElementById('content-' + agentMessageContainer.id);
                        // Replace with fully processed content that has markdown code blocks
                        renderFinalMessage(contentDiv, data.content);
                    }
                    break;
                case 'error':
                    if (agentMessageContainer) {
                        const contentDiv = document.getElementById('content-' + agentMessageContainer.id);
                        contentDiv.innerHTML = `<div class="text-red-500">${data.content}</div>`;
                    }
                    eventSource.close();
                    break;
                case 'end':
                    eventSource.close();
                    break;
            }
        };

        eventSource.onerror = function(event) {
            console.error('SSE Error:', event);
            const loadingIndicator = document.getElementById('loading-' + messageId);
            if (loadingIndicator) {
                loadingIndicator.innerHTML = '<div class="chat-message chat-message--agent"><div class="font-medium text-xs mb-1 text-gray-500">Stats Agent</div><div class="text-red-500 text-xs">Connection error</div></div>';
            }
            eventSource.close();
        };
    });
}

function processStreamingContent(contentDiv, content) {
    // Simple streaming - Go already converted tags to markdown code blocks
    contentDiv.innerHTML = marked.parse(content);

    // Apply syntax highlighting to any code blocks
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
}

function renderFinalMessage(contentDiv, content) {
    console.log('Processing content:', content);

    // Check if already processed to avoid recursion
    if (contentDiv.dataset.processed === 'true') {
        console.log('Content already processed, skipping');
        return;
    }

    // Mark as processed
    contentDiv.dataset.processed = 'true';

    // Go has already converted tags to markdown code blocks, so just render
    contentDiv.innerHTML = marked.parse(content);

    // Find and replace code blocks with collapsible components
    const pythonCodeBlocks = contentDiv.querySelectorAll('code.language-python');
    pythonCodeBlocks.forEach(block => {
        const code = block.textContent;
        const collapsible = createCollapsible('Python Code', code, 'python');
        const preElement = block.closest('pre');
        if (preElement) {
            preElement.replaceWith(collapsible);
        }
    });

    // Find regular code blocks that might be execution results (not Python)
    const regularCodeBlocks = contentDiv.querySelectorAll('pre > code:not(.language-python)');
    regularCodeBlocks.forEach(block => {
        const content = block.textContent;
        // Check if it looks like execution output (contains common output patterns)
        if (content.includes('Age') || content.includes('Mean') || content.includes('dtype') ||
            content.includes('columns') || content.includes('Missing') || content.includes('Saved:')) {
            const collapsible = createCollapsible('Execution Result', content, 'result');
            const preElement = block.closest('pre');
            if (preElement) {
                preElement.replaceWith(collapsible);
            }
        }
    });

    // Apply syntax highlighting to any remaining code
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
}

function createCollapsible(title, content, type) {
    const container = document.createElement('div');

    if (type === 'python') {
        container.className = 'my-6 rounded-2xl border border-gray-200 bg-gradient-to-br from-gray-50 to-slate-50 shadow-sm overflow-hidden animate-slide-up';
        container.innerHTML = `
            <div class="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-gray-100 to-slate-100 border-b border-gray-200/50 cursor-pointer hover:from-gray-200 hover:to-slate-200 transition-colors duration-200" onclick="toggleCodeBlock(this)">
                <div class="flex items-center space-x-3">
                    <svg class="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/>
                    </svg>
                    <span class="text-xs font-bold text-gray-700 uppercase tracking-wider font-mono">Python</span>
                </div>
                <span class="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded">Show</span>
            </div>
            <div class="code-content px-5 py-4 bg-white hidden">
                <pre class="font-mono text-sm overflow-x-auto"><code class="language-python">${content.trim()}</code></pre>
            </div>
        `;
    } else {
        container.className = 'my-6 rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50 to-green-50 shadow-sm overflow-hidden animate-slide-up';
        container.innerHTML = `
            <div class="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-emerald-100 to-green-100 border-b border-emerald-200/50 cursor-pointer hover:from-emerald-200 hover:to-green-200 transition-colors duration-200" onclick="toggleCodeBlock(this)">
                <div class="flex items-center space-x-3">
                    <svg class="w-3 h-3 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                    </svg>
                    <span class="text-xs font-bold text-emerald-700 uppercase tracking-wider font-mono">Execution Output</span>
                </div>
                <span class="text-xs font-medium text-emerald-600 bg-emerald-100 px-2 py-1 rounded">Show</span>
            </div>
            <div class="code-content px-5 py-4 bg-white hidden">
                <pre class="font-mono text-sm overflow-x-auto bg-gray-900 text-green-400 p-4 rounded-lg"><code>${content.trim()}</code></pre>
            </div>
        `;
    }
    return container;
}
