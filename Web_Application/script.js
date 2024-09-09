const converter = new showdown.Converter();

document.getElementById('send-button').addEventListener('click', function() {
    sendMessage();
});

document.getElementById('graph-button').addEventListener('click', function() {
    requestEquation();
});

document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('chat-input');

    textarea.addEventListener('input', function() {
        // Reset the height
        this.style.height = 'auto';
        
        // Set the height to match the scroll height
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Allow Shift+Enter to create a new line without submitting the form
    textarea.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('chat-input');

    // Adjust the height when typing
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Adjust the height when focused
    textarea.addEventListener('focus', function() {
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Reset the height when focus is lost
    textarea.addEventListener('blur', function() {
        this.style.height = '40px';  // Set this to your original min-height
    });

    // Allow Shift+Enter to create a new line without submitting the form
    textarea.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
});

function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const message = inputField.value.trim();
    
    if (message) {
        // Check if in equation mode
        if (inputField.getAttribute('data-mode') === 'equation') {
            graphEquation('user');
        } else {
            displayMessage(message, 'user');
            inputField.value = '';

            // Make API request to backend
            fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                displayTypingEffect(data.response, 'bot');
            })
            .catch(error => {
                console.error('Error:', error);
                displayMessage(`An error occurred: ${error.message}`, 'bot');
            });
        }
    }
}

function displayMessage(message, sender) {
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    
    // Convert Markdown to HTML
    const html = converter.makeHtml(message);
    
    // Apply the converted HTML to the message element
    messageElement.innerHTML = html;
    chatWindow.appendChild(messageElement);
  
    // Render KaTeX for any TeX expressions in the message
    renderMathInElement(messageElement, {
        delimiters: [
            {left: "$$", right: "$$", display: true},
            {left: "$", right: "$", display: false}
        ]
    });
  
    // Scroll to the bottom of the chat window
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function displayTypingEffect(message, sender, speed = 1, charsPerInterval = 1) { 
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    chatWindow.appendChild(messageElement);

    let index = 0;
    let rawText = '';

    function type() {
        if (index < message.length) {
            rawText += message.substring(index, index + charsPerInterval);
            messageElement.innerHTML = converter.makeHtml(rawText);
            index += charsPerInterval;
            setTimeout(type, speed);
        } else {
            messageElement.innerHTML = converter.makeHtml(rawText);
            renderMathInElement(messageElement, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false}
                ]
            });
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    }

    type();
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function requestEquation() {
    displayMessage("Please enter an equation to graph (e.g., y = 2x):", 'bot');
    document.getElementById('chat-input').setAttribute('data-mode', 'equation');
}

function graphEquation(sender) {
    const inputField = document.getElementById('chat-input');
    const equation = inputField.value.trim();
    
    if (equation) {
        displayMessage(`Graphing equation: ${equation}`, sender);
        inputField.value = '';

        // Create graph container
        const graphContainer = document.createElement('div');
        graphContainer.classList.add('graph-container', sender);
        const canvas = document.createElement('canvas');
        graphContainer.appendChild(canvas);

        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender, 'graph');
        messageElement.appendChild(graphContainer);
        document.getElementById('chat-window').appendChild(messageElement);

        // Scroll to the bottom of the chat window
        const chatWindow = document.getElementById('chat-window');
        chatWindow.scrollTop = chatWindow.scrollHeight;

        // Generate data points and create the graph
        try {
            const data = generateDataPoints(equation);
            new Chart(canvas, {
                type: 'line',
                data: {
                    labels: data.map(point => point.x),
                    datasets: [{
                        label: equation,
                        data: data.map(point => point.y),
                        borderColor: 'rgb(75, 192, 192)',
                        fill: false,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom'
                        },
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        } catch (error) {
            displayMessage(`Error: ${error.message}`, 'bot');
        }

        // Reset input mode
        inputField.removeAttribute('data-mode');
    }
}

function generateDataPoints(equation) {
    const points = [];
    const parsedEquation = equation.split('=');
    
    if (parsedEquation.length !== 2) {
        throw new Error("Invalid equation format. Use the format 'y = expression'.");
    }
    
    const expression = parsedEquation[1].trim();
    const compiledExpression = math.compile(expression);

    for (let x = -10; x <= 10; x += 0.25) {
        try {
            const y = compiledExpression.evaluate({x});
            points.push({x, y});
        } catch (error) {
            console.error('Error calculating point:', error);
            throw new Error("Failed to calculate points for the equation.");
        }
    }

    return points;
}

