// Auto-scroll to bottom when new messages are added
function observeMessages() {
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        const observer = new MutationObserver(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        });
        observer.observe(messagesContainer, { childList: true, subtree: true });
    }
}

// Focus input on page load
function focusInput() {
    const messageInput = document.getElementById('message-input');
    if (messageInput && document.activeElement !== messageInput) {
        messageInput.focus();
    }
}

// New function to handle copying code to the clipboard with a fallback
function copyCode(button) {
    const codeBlock = button.closest('.my-4').querySelector('code');
    const textToCopy = codeBlock.textContent;
    const copyText = button.querySelector('.copy-text');

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyText.textContent = 'Copied!';
            setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    } else {
        // Fallback for non-secure contexts (HTTP)
        const textArea = document.createElement("textarea");
        textArea.value = textToCopy;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        
        // Use the preventScroll option to stop the page from jumping
        textArea.focus({ preventScroll: true }); 
        textArea.select();

        try {
            document.execCommand('copy');
            copyText.textContent = 'Copied!';
            setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
        } catch (err) {
            console.error('Fallback copy failed: ', err);
        }
        document.body.removeChild(textArea);
    }
}

// Update toggleCodeBlock to use the parent container
function toggleCodeBlock(element) {
    const header = element.closest('.flex.items-center.justify-between');
    const content = header.nextElementSibling;
    const icon = header.querySelector('.chevron-icon');
    const actionText = header.querySelector('.action-text');
    
    content.classList.toggle('hidden');
    icon.classList.toggle('rotate-180');
    
    if (content.classList.contains('hidden')) {
        actionText.textContent = 'Show';
    } else {
        actionText.textContent = 'Hide';
    }
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
        let debounceTimer;

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
                            <div class="agent-output bg-white rounded-2xl px-5 py-3 max-w-2xl shadow-md border border-gray-100">
                                <div class="font-semibold text-sm text-primary mb-2 font-display">Stats Agent</div>
                                <div id="content-${agentMessageId}" class="prose prose-sm max-w-none leading-relaxed text-gray-700 font-sans"></div>
                            </div>
                        </div>
                    `;
                    document.getElementById('messages').appendChild(agentMessageContainer);
                    break;
                case 'chunk':
                    if (agentMessageContainer && typeof data.content === 'string') {
                        contentBuffer += data.content;
                        
                        clearTimeout(debounceTimer);
                        debounceTimer = setTimeout(() => {
                            const contentDiv = document.getElementById('content-' + agentMessageContainer.id);
                            if(contentDiv){
                                renderAndProcessContent(contentDiv, contentBuffer);
                            }
                        }, 50);
                    }
                    break;
                case 'end':
                    eventSource.close();
                    if (agentMessageContainer) {
                        const contentDiv = document.getElementById('content-' + agentMessageContainer.id);
                        if (contentDiv) {
                           renderAndProcessContent(contentDiv, contentBuffer);
                        }
                    }
                    break;
                default:
                    break;
            }
        };

        eventSource.onerror = function(event) {
            console.error('SSE Error:', event);
            const loadingIndicator = document.getElementById('loading-' + messageId);
            if (loadingIndicator) {
                loadingIndicator.innerHTML = '<div class="text-red-500 text-xs">Connection error</div>';
            }
            eventSource.close();
        };
    });
}

function renderAndProcessContent(contentDiv, content) {
    const cleanedContent = content.replace(/Agent:\s*/g, '').trim();
    if (cleanedContent === "undefined") return;
    
    contentDiv.innerHTML = marked.parse(cleanedContent || '');

    contentDiv.querySelectorAll('pre > code:not([data-collapsible="true"])').forEach(block => {
        const preElement = block.parentElement;
        block.setAttribute('data-collapsible', 'true');

        const codeContent = block.textContent;
        let collapsible;

        if (block.classList.contains('language-python')) {
            collapsible = createCollapsible(codeContent, 'python');
        } else {
            collapsible = createCollapsible(codeContent, 'result');
        }
        
        preElement.replaceWith(collapsible);
        if (typeof hljs !== 'undefined') {
             collapsible.querySelectorAll('pre code').forEach(newBlock => {
                hljs.highlightElement(newBlock);
            });
        }
    });
}

function createCollapsible(content, type) {
    let template;
    if (type === 'python') {
        template = document.getElementById('python-block-template');
    } else {
        template = document.getElementById('execution-block-template');
    }

    const container = template.cloneNode(true);
    container.id = '';

    const codeElement = container.querySelector('code');
    codeElement.textContent = content.trim();

    return container;
}