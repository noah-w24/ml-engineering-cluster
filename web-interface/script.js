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

  detectBtn.addEventListener("click", () => {
    const text = textarea.value;
    console.log("Submitting for violence detection:", text);

    // Logic to call the API will go here
    alert(`Submit logic to be implemented.\n\nText: "${text}"`);
  });
});
