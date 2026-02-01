document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("tweet-text");
  const charCountSpan = document.getElementById("char-count");
  const detectBtn = document.getElementById("detect-btn");
  const progressCircle = document.querySelector(".progress-ring__circle");

  const MAX_CHARS = 280;
  const RADIUS = 8;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

  // Initialize circular progress
  progressCircle.style.strokeDasharray = `${CIRCUMFERENCE} ${CIRCUMFERENCE}`;
  progressCircle.style.strokeDashoffset = CIRCUMFERENCE;

  function setProgress(percent) {
    const offset = CIRCUMFERENCE - (percent / 100) * CIRCUMFERENCE;
    progressCircle.style.strokeDashoffset = offset;
  }

  // Auto-resize textarea
  textarea.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";

    const currentLength = this.value.length;
    charCountSpan.textContent = currentLength;

    // Update Button State
    if (currentLength > 0) {
      detectBtn.removeAttribute("disabled");
    } else {
      detectBtn.setAttribute("disabled", "true");
    }

    // Update Progress Ring
    const percent = Math.min((currentLength / MAX_CHARS) * 100, 100);
    setProgress(percent);

    // Visual cues for nearing limit
    if (currentLength >= MAX_CHARS) {
      progressCircle.style.stroke = "#f4212e"; // Red
      charCountSpan.style.color = "#f4212e";
    } else if (currentLength >= MAX_CHARS - 20) {
      progressCircle.style.stroke = "#ffd400"; // Yellow
      charCountSpan.style.color = "#ffd400";
    } else {
      progressCircle.style.stroke = "#1d9bf0"; // Blue
      charCountSpan.style.color = "#71767b";
    }
  });

  detectBtn.addEventListener('click', async () => {
        const text = textarea.value;
        const resultArea = document.getElementById('result-area');
        
        // Indicate loading
        detectBtn.disabled = true;
        detectBtn.textContent = "Analyzing...";
        resultArea.classList.add('hidden');
        resultArea.innerHTML = '';

        try {
            const response = await fetch('https://backend.group-avgk.dski23a.timebertt.dev/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display Results
            resultArea.classList.remove('hidden');
            
            if (data.predictions && data.predictions.length > 0) {
                let html = '<h3>Analysis Results</h3><div class="results-grid">';
                data.predictions.forEach(p => {
                    const isViolent = p.prediction === '1' || p.prediction === 'violent' || p.prediction === 'True';
                    const colorClass = isViolent ? 'violent' : 'safe';
                    html += `
                        <div class="result-card ${colorClass}">
                            <div class="model-name">${p.model_name.length > 8 ? p.model_name.substring(0, 8) + '...' : p.model_name}</div>
                            <div class="prediction">${isViolent ? 'VIOLENT' : 'NON-VIOLENT'}</div>
                            <div class="confidence">Confidence: ${(p.confidence * 100).toFixed(1)}%</div>
                        </div>
                    `;
                });
                html += '</div>';
                resultArea.innerHTML = html;
            } else {
                resultArea.innerHTML = '<p>No predictions returned.</p>';
            }

        } catch (error) {
            console.error('Error:', error);
            resultArea.classList.remove('hidden');
            resultArea.innerHTML = `<p style="color: var(--danger-color)">Error analyzing text: ${error.message}</p>`;
        } finally {
            detectBtn.disabled = false;
            detectBtn.textContent = "Detect Violence";
        }
    });
});
