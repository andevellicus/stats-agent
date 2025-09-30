let activeEventSource = null;
let autoScrollEnabled = true;

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
        fileBadgeContainer.innerHTML = `
            <div id="file-badge" class="inline-flex items-center justify-between bg-blue-100 text-blue-800 text-sm font-semibold px-3 py-2 rounded-lg mb-2 border border-blue-200 animate-fade-in">
                <div class="flex items-center space-x-2">
                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a2 2 0 00-2 2v8a2 2 0 002 2h2a2 2 0 002-2V4a2 2 0 00-2-2H9z"></path><path fill-rule="evenodd" d="M4 8a2 2 0 012-2h3v10H6a2 2 0 01-2-2V8zm12 0a2 2 0 00-2-2h-3v10h3a2 2 0 002-2V8z" clip-rule="evenodd"></path></svg>
                    <span>${fileName}</span>
                </div>
                <button type="button" id="remove-file-btn" class="text-blue-500 hover:text-blue-700">
                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>
                </button>
            </div>
        `;

        document.getElementById('remove-file-btn').addEventListener('click', () => {
            fileInput.value = ''; // Clear the file input
            fileBadgeContainer.innerHTML = ''; // Remove the badge
            form.removeAttribute('enctype'); // Reset form encoding
        });
    }


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
    autoScrollEnabled = false;
    const codeContainer = button.closest('.overflow-hidden');
    if (!codeContainer) {
        console.error("Could not find the code container to copy from.");
        autoScrollEnabled = true;
        return;
    }
    const codeBlock = codeContainer.querySelector('code');
    const textToCopy = codeBlock.textContent;
    const copyText = button.querySelector('.copy-text');

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyText.textContent = 'Copied!';
            setTimeout(() => {
                copyText.textContent = 'Copy';
                autoScrollEnabled = true;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            autoScrollEnabled = true;
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
            setTimeout(() => {
                copyText.textContent = 'Copy';
                autoScrollEnabled = true;
            }, 2000);
        } catch (err) {
            console.error('Fallback copy failed: ', err);
            autoScrollEnabled = true;
        }
        document.body.removeChild(textArea);
    }
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
    const cleanedContent = content.replace(/Agent:\s*/g, '').trim();
    if (cleanedContent === "undefined") return;

    contentDiv.innerHTML = marked.parse(cleanedContent || '');

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

        // Check if the previous element was a Python code block wrapper
        const previousElement = preElement.previousElementSibling;
        if (!isPython && previousElement && previousElement.id === 'python-block-template') {
             // If it is, apply the 'dock-top' class to our new wrapper
             wrapper.classList.add('dock-top');
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

    // Handle agent status messages (this logic is unchanged)
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