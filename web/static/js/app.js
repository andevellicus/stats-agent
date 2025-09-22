let activeEventSource = null;

function autoExpand(textarea) {
    textarea.style.height = 'auto'; // Reset height
    textarea.style.height = (textarea.scrollHeight) + 'px'; // Set to scroll height
}

// This function will be attached to the MutationObserver
function setupAutoScroll() {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    // This function scrolls the container to the bottom
    const scrollToBottom = () => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    };

    // Create an observer instance linked to a callback function
    const observer = new MutationObserver(scrollToBottom);

    // Start observing the target node for configured mutations
    observer.observe(messagesContainer, {
        childList: true, // observe direct children additions/removals
        subtree: true,   // observe all descendants
        characterData: true // observe text changes
    });

    // Initial scroll to bottom
    scrollToBottom();
}

function setupFormListener() {
    const form = document.getElementById('chat-form');
    const submitButton = document.getElementById('submit-button');
    const sendIcon = document.getElementById('send-icon');
    const stopIcon = document.getElementById('stop-icon');
    const messageInput = document.getElementById('message-input');

    if (!form) return;

    submitButton.addEventListener('click', (event) => {
        if (activeEventSource) {
            event.preventDefault();
            activeEventSource.close();
            console.log("SSE connection closed by user.");
        }
    });

    form.addEventListener('htmx:beforeRequest', () => {
        sendIcon.classList.add('hidden');
        stopIcon.classList.remove('hidden');
        messageInput.disabled = true;
    });

    form.addEventListener('htmx:afterRequest', () => {
        messageInput.style.height = 'auto'; // Reset textarea height
    });
}

function focusInput() {
    const messageInput = document.getElementById('message-input');
    if (messageInput && document.activeElement !== messageInput) {
        messageInput.focus();
    }
}

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
        const textArea = document.createElement("textarea");
        textArea.value = textToCopy;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
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

function submitOnEnter(event) {
    if (event.keyCode == 13 && !event.shiftKey) {
        event.preventDefault();
        event.target.form.requestSubmit();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    focusInput();
    initiateSSE();
    setupFormListener();
    setupAutoScroll(); // Set up the observer when the page loads

    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('input', () => autoExpand(messageInput));
    }
});

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
        
        activeEventSource = eventSource;

        let contentBuffer = '';
        let agentMessageContainer = null;
        let debounceTimer;

        const cleanup = () => {
            const sendIcon = document.getElementById('send-icon');
            const stopIcon = document.getElementById('stop-icon');
            const messageInput = document.getElementById('message-input');
            
            if (sendIcon && stopIcon) {
                stopIcon.classList.add('hidden');
                sendIcon.classList.remove('hidden');
            }
            if(messageInput) {
                messageInput.disabled = false;
            }
            activeEventSource = null;
        };

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
                    cleanup();
                    break;
                default:
                    break;
            }
        };

        eventSource.onerror = function(event) {
            console.error('SSE Error:', event);
            cleanup();
            eventSource.close();
        };
    });
}

function renderAndProcessContent(contentDiv, content) {
    const cleanedContent = content.replace(/Agent:\s*/g, '').trim();
    if (cleanedContent === "undefined") return;
    
    contentDiv.innerHTML = marked.parse(cleanedContent || '');

    contentDiv.querySelectorAll('.agent-status-message').forEach(statusElement => {
        const statusText = statusElement.textContent;
        const template = document.getElementById('agent-status-template');
        if (template) {
            const statusClone = template.cloneNode(true);
            statusClone.id = '';
            statusClone.querySelector('span').textContent = statusText;
            statusElement.replaceWith(statusClone);
        }
    });

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