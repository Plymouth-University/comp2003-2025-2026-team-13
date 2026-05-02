
async function sendToPython(data) {

    const response = await fetch("/process_frame", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    console.log("Python response:", result);
}
