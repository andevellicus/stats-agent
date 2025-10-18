let activeEventSource = null;
let autoScrollEnabled = true;

// Toggle sidebar visibility on mobile
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');

    if (!sidebar || !backdrop) return;

    const isOpen = sidebar.classList.contains('translate-x-0');

    if (isOpen) {
        // Close sidebar
        sidebar.classList.remove('translate-x-0');
        sidebar.classList.add('-translate-x-full');
        backdrop.classList.add('hidden');
        document.body.style.overflow = ''; // Re-enable body scroll
    } else {
        // Open sidebar
        sidebar.classList.remove('-translate-x-full');
        sidebar.classList.add('translate-x-0');
        backdrop.classList.remove('hidden');
        document.body.style.overflow = 'hidden'; // Prevent body scroll on mobile when sidebar is open
    }
}

function applySyntaxHighlighting() {
    if (typeof hljs !== 'undefined') {
        document.querySelectorAll('pre code.language-python:not(.hljs)').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
}

function resetChatForm(form) {
    if (form) {
        form.reset();
    }
    const fileBadgeContainer = document.getElementById('file-upload-badge-container');
    if (fileBadgeContainer) {
        fileBadgeContainer.innerHTML = '';
    }
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.value = '';
    }
    if (form) {
        form.removeAttribute('enctype');
    }
}

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
        if (autoScrollEnabled) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
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
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const fileBadgeContainer = document.getElementById('file-upload-badge-container'); // Get the container

    if (!form) return;

    // Check if already initialized to prevent duplicate listeners
    if (form.hasAttribute('data-listeners-initialized')) return;
    form.setAttribute('data-listeners-initialized', 'true');

    uploadButton.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // A file is selected
            form.setAttribute('enctype', 'multipart/form-data');
            renderFileBadge(file.name);
        } else {
            // A file was removed or selection was cancelled
            fileBadgeContainer.innerHTML = '';
            form.removeAttribute('enctype');
        }
    });

    function renderFileBadge(fileName) {
        const template = document.getElementById('file-badge-template');
        if (!template) {
            console.error('File badge template not found!');
            return;
        }

        // Clone the template's content
        const newBadge = template.querySelector('#file-badge').cloneNode(true);
        
        // Update the filename in the clone
        const fileNameSpan = newBadge.querySelector('.file-name');
        if (fileNameSpan) {
            fileNameSpan.textContent = fileName;
        }

        // Add the new badge to the container
        const fileBadgeContainer = document.getElementById('file-upload-badge-container');
        fileBadgeContainer.innerHTML = ''; // Clear previous badges
        fileBadgeContainer.appendChild(newBadge);

        // Add event listener to the new remove button
        newBadge.querySelector('.remove-file-btn').addEventListener('click', () => {
            const fileInput = document.getElementById('file-input');
            const form = document.getElementById('chat-form');
            
            fileInput.value = ''; // Clear the file input
            fileBadgeContainer.innerHTML = ''; // Remove the badge
            form.removeAttribute('enctype'); // Reset form encoding
        });
    }


    submitButton.addEventListener('click', (event) => {
        if (activeEventSource) {
            event.preventDefault();

            // Get session ID from the form
            const sessionIdInput = form.querySelector('input[name="session_id"]');
            const sessionId = sessionIdInput ? sessionIdInput.value : null;

            // Call stop endpoint to cancel agent execution
            if (sessionId) {
                fetch(`/chat/stop?session_id=${encodeURIComponent(sessionId)}`, {
                    method: 'POST'
                }).then(() => {
                    console.log("Agent execution stopped by user.");
                }).catch(err => {
                    console.error("Failed to stop agent execution:", err);
                });
            }

            // Close SSE connection
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
    autoScrollEnabled = false;
    const codeContainer = button.closest('.overflow-hidden');
    if (!codeContainer) {
        console.error("Could not find the code container to copy from.");
        autoScrollEnabled = true; // Re-enable on error
        return;
    }
    const codeBlock = codeContainer.querySelector('code');
    const textToCopy = codeBlock.textContent;
    const copyText = button.querySelector('.copy-text');

    const handleCopySuccess = () => {
        copyText.textContent = 'Copied!';
        // Separate timer just for UI feedback, does not affect scrolling
        setTimeout(() => {
            copyText.textContent = 'Copy';
        }, 1000);
    };

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(textToCopy).then(handleCopySuccess).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    } else {
        // Fallback for non-secure contexts
        const textArea = document.createElement("textarea");
        textArea.value = textToCopy;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus({ preventScroll: true });
        textArea.select();
        try {
            document.execCommand('copy');
            handleCopySuccess();
        } catch (err) {
            console.error('Fallback copy failed: ', err);
        }
        document.body.removeChild(textArea);
    }

    // Re-enable scrolling after a very short delay
    setTimeout(() => {
        autoScrollEnabled = true;
    }, 1100);
}

function toggleCodeBlock(element) {
    autoScrollEnabled = false;
    const header = element.closest('.flex.items-center.justify-between');
    const content = header.nextElementSibling;
    const icon = element.querySelector('.chevron-icon');
    const actionText = element.querySelector('.action-text');

    content.classList.toggle('hidden');
    icon.classList.toggle('rotate-180');

    if (content.classList.contains('hidden')) {
        actionText.textContent = 'Show';
    } else {
        actionText.textContent = 'Hide';
    }
    setTimeout(() => {
        autoScrollEnabled = true;
    }, 100);
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
    applySyntaxHighlighting(); // Apply on initial page load

    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('input', () => autoExpand(messageInput));
    }

    // After initial render, try to reattach to any active run
    checkAndAttachToActiveRun();
});

// Listen for HTMX event to update session ID after it's created
document.body.addEventListener('updateSessionId', function(event) {
    const form = document.getElementById('chat-form');
    if (!form) return;

    const sessionInput = form.querySelector('input[name="session_id"]');
    if (sessionInput && event.detail && event.detail.value) {
        const newSessionId = event.detail.value;
        sessionInput.value = newSessionId;
        console.log('Session ID updated from placeholder to:', newSessionId);
    }
});

document.body.addEventListener('htmx:afterSwap', function(event) {
    focusInput();
    initiateSSE();
    setupFormListener(); // Re-attach form event listeners after htmx loads new content
    setupAutoScroll(); // Re-setup autoscroll for the messages container
    applySyntaxHighlighting(); // Re-apply after htmx loads new content

    // Re-attach textarea auto-expand listener
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('input', () => autoExpand(messageInput));
    }

    // Close sidebar on mobile after navigation
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    if (sidebar && backdrop && window.innerWidth < 768) {
        sidebar.classList.remove('translate-x-0');
        sidebar.classList.add('-translate-x-full');
        backdrop.classList.add('hidden');
        document.body.style.overflow = '';
    }

    // Try to reattach to any active run after navigation
    checkAndAttachToActiveRun();
});

function checkAndAttachToActiveRun() {
    // If already streaming, do nothing
    if (activeEventSource) return;
    const form = document.getElementById('chat-form');
    if (!form) return;
    const sessionIdInput = form.querySelector('input[name="session_id"]');
    const sessionId = sessionIdInput ? sessionIdInput.value : null;
    if (!sessionId) return;

    fetch(`/chat/status?session_id=${encodeURIComponent(sessionId)}`, { method: 'GET' })
        .then(resp => resp.ok ? resp.json() : null)
        .then(data => {
            if (!data || !data.running || !data.user_message_id) return;
            // Toggle UI to show stop available
            const sendIcon = document.getElementById('send-icon');
            const stopIcon = document.getElementById('stop-icon');
            const messageInput = document.getElementById('message-input');
            if (sendIcon && stopIcon) {
                sendIcon.classList.add('hidden');
                stopIcon.classList.remove('hidden');
            }
            if (messageInput) { messageInput.disabled = true; }
            // Attach SSE to the active stream
            attachSSE(sessionId, data.user_message_id);
        })
        .catch(() => {});
}

function attachSSE(sessionId, messageId) {
    if (activeEventSource) return;
    const eventSource = new EventSource('/chat/stream?session_id=' + encodeURIComponent(sessionId) + '&user_message_id=' + encodeURIComponent(messageId));
    activeEventSource = eventSource;

    let contentBuffer = '';
    let messageContainer = null;
    let debounceTimer;

    const cleanup = () => {
        const sendIcon = document.getElementById('send-icon');
        const stopIcon = document.getElementById('stop-icon');
        const messageInput = document.getElementById('message-input');
        if (sendIcon && stopIcon) {
            stopIcon.classList.add('hidden');
            sendIcon.classList.remove('hidden');
        }
        if(messageInput) { messageInput.disabled = false; }
        activeEventSource = null;
    };

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case 'connection_established':
                break;
            case 'remove_loader':
                const loadingIndicator = document.getElementById(data.content);
                if (loadingIndicator) { loadingIndicator.remove(); }
                break;
            case 'create_container':
                const agentMessageId = 'agent-msg-' + data.content;
                messageContainer = document.createElement('div');
                messageContainer.id = agentMessageId;
                messageContainer.className = "w-full";
                messageContainer.innerHTML = `
                    <div class=\"bg-white rounded-2xl px-5 py-3 w-full shadow-md border border-gray-100 hover:shadow-lg transition-shadow duration-200\">\n
                        <div class=\"font-semibold text-sm text-primary mb-2 font-display\">Pocket Statistician</div>\n
                        <div id=\"content-${agentMessageId}\" class=\"prose max-w-none leading-relaxed text-gray-700 font-sans\"></div>\n
                        <div id=\"file-container-${agentMessageId}\"></div>\n
                    </div>`;
                document.getElementById('messages').appendChild(messageContainer);
                break;
            case 'sidebar_update':
                if (data.content) {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(data.content, 'text/html');
                    const newLink = doc.body.firstChild;
                    if (newLink) {
                        const targetId = newLink.id;
                        const targetLink = document.getElementById(targetId);
                        if (targetLink) { targetLink.innerHTML = newLink.innerHTML; }
                    }
                }
                break;
            case 'file_append_html':
                if (data.content) {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(data.content, 'text/html');
                    const fileContainer = doc.body.firstChild;
                    if (fileContainer) {
                        const targetId = fileContainer.id;
                        const targetDiv = document.getElementById(targetId);
                        if (targetDiv) { targetDiv.innerHTML = fileContainer.innerHTML; }
                    }
                }
                break;
            case 'chunk':
                if (messageContainer && typeof data.content === 'string') {
                    contentBuffer += data.content;
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(() => {
                        const contentDiv = document.getElementById('content-' + messageContainer.id);
                        if(contentDiv){ renderAndProcessContent(contentDiv, contentBuffer); }
                    }, 50);
                }
                break;
            case 'end':
                eventSource.close();
                if (messageContainer) {
                    const contentDiv = document.getElementById('content-' + messageContainer.id);
                    if (contentDiv) { renderAndProcessContent(contentDiv, contentBuffer); }
                }
                cleanup();
                break;
            default:
                break;
        }
    };

    eventSource.onerror = function(event) {
        cleanup();
        eventSource.close();
    };
}

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
        let messageContainer = null;
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
                    messageContainer = document.createElement('div');
                    messageContainer.id = agentMessageId;
                    messageContainer.className = "w-full";
                    messageContainer.innerHTML = `
                        <div class="bg-white rounded-2xl px-5 py-3 w-full shadow-md border border-gray-100 hover:shadow-lg transition-shadow duration-200">
                            <div class="font-semibold text-sm text-primary mb-2 font-display">Pocket Statistician</div>
                            <div id="content-${agentMessageId}" class="prose max-w-none leading-relaxed text-gray-700 font-sans"></div>
                            <div id="file-container-${agentMessageId}"></div>
                        </div>
                    `;
                    document.getElementById('messages').appendChild(messageContainer);
                    break;
                 case 'sidebar_update':
                    if (data.content) {
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(data.content, 'text/html');
                        const newLink = doc.body.firstChild;
                        if (newLink) {
                            const targetId = newLink.id;
                            const targetLink = document.getElementById(targetId);
                            if (targetLink) {
                                targetLink.innerHTML = newLink.innerHTML;
                            }
                        }
                    }
                    break;
                 case 'file_append_html':
                    if (data.content) {
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(data.content, 'text/html');
                        const fileContainer = doc.body.firstChild;
                        if (fileContainer) {
                            const targetId = fileContainer.id;
                            const targetDiv = document.getElementById(targetId);
                            if (targetDiv) {
                                targetDiv.innerHTML = fileContainer.innerHTML;
                            }
                        }
                    }
                    break;
                case 'chunk':
                    if (messageContainer && typeof data.content === 'string') {
                        contentBuffer += data.content;

                        clearTimeout(debounceTimer);
                        debounceTimer = setTimeout(() => {
                            const contentDiv = document.getElementById('content-' + messageContainer.id);
                            if(contentDiv){
                                renderAndProcessContent(contentDiv, contentBuffer);
                            }
                        }, 50);
                    }
                    break;
                case 'end':
                    eventSource.close();
                    if (messageContainer) {
                        const contentDiv = document.getElementById('content-' + messageContainer.id);
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
    // Normalize content to fix encoding issues that can break marked.js parsing
    let normalized = content || '';

    // Fix common issues:
    // 1. Unescape HTML entities for backticks (can happen during SSE transmission)
    normalized = normalized.replace(/&#96;/g, '`');
    // 2. Fix any escaped backticks
    normalized = normalized.replace(/\\`/g, '`');

    contentDiv.innerHTML = marked.parse(normalized);

    // Process all <pre> blocks generated by marked.js
    contentDiv.querySelectorAll('pre:not([data-collapsible="true"])').forEach((preElement, index) => {
        preElement.setAttribute('data-collapsible', 'true');

        const codeElement = preElement.querySelector('code');
        if (!codeElement) return;

        const isPython = codeElement.classList.contains('language-python');
        const componentType = isPython ? 'python' : 'result';
        const templateId = isPython ? 'python-block-template' : 'execution-block-template';
        const template = document.getElementById(templateId);
        if (!template) return;
        
        // Create the new component wrapper from the template
        const wrapper = template.cloneNode(true);
        wrapper.id = '';
        
        // Find the code element within the new wrapper and update its content
        const newCodeElement = wrapper.querySelector('code');
        if (newCodeElement) {
            newCodeElement.textContent = codeElement.textContent;
        }

        preElement.replaceWith(wrapper);

        // Apply syntax highlighting
        if (typeof hljs !== 'undefined' && isPython) {
            const blockToHighlight = wrapper.querySelector('code');
            delete blockToHighlight.dataset.highlighted;
            blockToHighlight.classList.remove('hljs');
            hljs.highlightElement(blockToHighlight);
        }
    });

    // Handle agent status messages - both raw <agent_status> tags and .agent-status-message divs
    // First, handle raw <agent_status> tags from streaming
    contentDiv.querySelectorAll('agent_status').forEach(statusElement => {
        const statusText = statusElement.textContent;
        const template = document.getElementById('agent-status-template');
        if (template) {
            const statusClone = template.cloneNode(true);
            statusClone.id = '';
            statusClone.querySelector('span').textContent = statusText;
            statusElement.replaceWith(statusClone);
        }
    });

    // Then, handle .agent-status-message divs from database rendering
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
}
