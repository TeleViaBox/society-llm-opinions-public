let chatHistoryIds = null;

async function sendMessage() {
    const inputField = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");
    const userMessage = inputField.value;
    inputField.value = "";
    inputField.focus();

    // Update the chat box with the user's message
    chatBox.innerHTML += `<div>User: ${userMessage}</div>`;

    // Send the message to the Flask backend
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage, chat_history_ids: chatHistoryIds }),
    });

    const data = await response.json();
    chatHistoryIds = data.chat_history_ids;

    // Update the chat box with the bot's response
    chatBox.innerHTML += `<div>DialoGPT: ${data.response}</div>`;
    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}
