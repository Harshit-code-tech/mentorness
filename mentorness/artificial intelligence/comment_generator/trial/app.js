document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles.js with the configuration
    particlesJS.load('particles-js', 'scripts/particles-config.json', function() {
        console.log('particles.js loaded - callback');
    });

    // Form submission handler
    document.getElementById('analyzeButton').addEventListener('click', function() {
        const content = document.getElementById('content').value;
        analyzeContent(content);
    });

    function analyzeContent(content) {
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ content: content })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('error').innerText = data.error;
                return;
            }
            document.getElementById('error').innerText = '';
            document.getElementById('friendly').innerText = data.friendly;
            document.getElementById('funny').innerText = data.funny;
            document.getElementById('congratulating').innerText = data.congratulating;
            document.getElementById('questioning').innerText = data.questioning;
            document.getElementById('disagreement').innerText = data.disagreement;
        })
        .catch(error => {
            document.getElementById('error').innerText = 'An error occurred';
            console.error('Error:', error);
        });
    }
});
