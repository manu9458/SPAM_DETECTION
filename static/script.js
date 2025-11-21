document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('messageInput').value;
    if (!text.trim()) return;

    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const loader = document.getElementById('loader');
    const predictionTitle = document.getElementById('predictionTitle');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');

    // Reset UI
    resultContainer.classList.remove('hidden');
    resultContent.classList.add('hidden');
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Update UI with results
        loader.classList.add('hidden');
        resultContent.classList.remove('hidden');

        const isSpam = data.prediction.toLowerCase() === 'spam';
        const percentage = (data.probability * 100).toFixed(1);

        predictionTitle.textContent = isSpam ? 'SPAM DETECTED' : 'LOOKS SAFE';
        predictionTitle.className = isSpam ? 'spam-result' : 'ham-result';

        confidenceFill.style.width = `${percentage}%`;
        confidenceFill.style.backgroundColor = isSpam ? 'var(--danger)' : 'var(--success)';
        
        confidenceText.textContent = `Confidence: ${percentage}%`;

    } catch (error) {
        console.error('Error:', error);
        loader.classList.add('hidden');
        alert('Something went wrong. Please try again.');
    }
});
