/* Chat sidebar WebSocket client */

let chatWs = null;
let chatSessionId = null;
let reconnectTimer = null;

function generateSessionId() {
    return 'web-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
}

function openChat(context, id) {
    const sidebar = document.getElementById('chat-sidebar');
    if (!sidebar) return;

    sidebar.classList.remove('hidden');
    document.body.classList.add('chat-open');

    // Set title based on context
    const titleEl = document.getElementById('chat-title');
    if (context && id) {
        const labels = {signal: 'Signal', lead: 'Lead', recommendation: 'Recommendation'};
        titleEl.textContent = 'Discuss ' + (labels[context] || context) + ' #' + id;
    } else {
        titleEl.textContent = 'Chat with Nano';
    }

    // Clear previous messages
    const messagesEl = document.getElementById('chat-messages');
    messagesEl.innerHTML = '';

    // Connect WebSocket
    chatSessionId = generateSessionId();
    connectWebSocket(context, id);
}

function closeSidebar() {
    const sidebar = document.getElementById('chat-sidebar');
    if (sidebar) {
        sidebar.classList.add('hidden');
        document.body.classList.remove('chat-open');
    }
    disconnectWebSocket();
}

function connectWebSocket(context, id) {
    disconnectWebSocket();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let url = protocol + '//' + window.location.host + '/chat/ws/' + chatSessionId;

    if (context && id) {
        url += '?context=' + encodeURIComponent(context) + '&id=' + encodeURIComponent(id);
    }

    chatWs = new WebSocket(url);

    chatWs.onopen = function() {
        clearTimeout(reconnectTimer);
        appendStatus('Connected');
    };

    chatWs.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'message') {
                appendMessage('nano', data.content);
                hideTyping();
            } else if (data.type === 'progress') {
                showTyping(data.detail || data.tool_name || 'Thinking...');
            } else if (data.type === 'error') {
                appendStatus('Error: ' + data.content);
            }
        } catch (e) {
            appendMessage('nano', event.data);
        }
    };

    chatWs.onclose = function() {
        appendStatus('Disconnected');
        // Auto-reconnect after 3 seconds
        reconnectTimer = setTimeout(function() {
            if (!document.getElementById('chat-sidebar').classList.contains('hidden')) {
                connectWebSocket(context, id);
            }
        }, 3000);
    };

    chatWs.onerror = function() {
        appendStatus('Connection error');
    };
}

function disconnectWebSocket() {
    clearTimeout(reconnectTimer);
    if (chatWs) {
        chatWs.close();
        chatWs = null;
    }
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const content = input.value.trim();
    if (!content || !chatWs || chatWs.readyState !== WebSocket.OPEN) return;

    appendMessage('user', content);
    chatWs.send(JSON.stringify({content: content}));
    input.value = '';
    showTyping('Thinking...');
}

function appendMessage(role, content) {
    const messagesEl = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg chat-msg-' + role;

    const labelEl = document.createElement('div');
    labelEl.className = 'chat-msg-label';
    labelEl.textContent = role === 'user' ? 'You' : 'Nano';

    const contentEl = document.createElement('div');
    contentEl.className = 'chat-msg-content';
    contentEl.textContent = content;

    msgDiv.appendChild(labelEl);
    msgDiv.appendChild(contentEl);
    messagesEl.appendChild(msgDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendStatus(text) {
    const messagesEl = document.getElementById('chat-messages');
    const statusEl = document.createElement('div');
    statusEl.className = 'chat-status';
    statusEl.textContent = text;
    messagesEl.appendChild(statusEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping(detail) {
    let typingEl = document.getElementById('chat-typing');
    if (!typingEl) {
        typingEl = document.createElement('div');
        typingEl.id = 'chat-typing';
        typingEl.className = 'chat-typing';
        document.getElementById('chat-messages').appendChild(typingEl);
    }
    typingEl.textContent = detail;
    typingEl.style.display = 'block';
    document.getElementById('chat-messages').scrollTop =
        document.getElementById('chat-messages').scrollHeight;
}

function hideTyping() {
    const typingEl = document.getElementById('chat-typing');
    if (typingEl) {
        typingEl.style.display = 'none';
    }
}

// Handle Enter key in textarea
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('chat-input');
    if (input) {
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
});
