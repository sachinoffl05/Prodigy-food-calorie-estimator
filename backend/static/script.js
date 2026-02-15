let dailyCalories = 0;

function sendImage() {
    const input = document.getElementById("imageInput");
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
    const bar = document.getElementById("confidenceBar");
    const daily = document.getElementById("dailyTotal");
    const preview = document.getElementById("preview");

    if (input.files.length === 0) {
        alert("Please select an image");
        return;
    }

    // Show image preview
    const file = input.files[0];
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    // Reset UI
    loading.style.display = "block";
    result.innerHTML = "";
    bar.style.width = "0%";

    const formData = new FormData();
    formData.append("image", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loading.style.display = "none";

        result.innerHTML = `
            <p><b>🍕 Food:</b> ${data.food}</p>
            <p><b>📊 Confidence:</b> ${data.confidence}%</p>
            <p><b>🔥 Calories:</b> ${data.calories} kcal</p>
        `;

        bar.style.width = data.confidence + "%";

        dailyCalories += data.calories;
        daily.innerText = "Today's Calories: " + dailyCalories + " kcal";
    })
    .catch(() => {
        loading.style.display = "none";
        result.innerText = "❌ Prediction failed";
    });
}
