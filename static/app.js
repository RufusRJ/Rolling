// AeroVision Borescope Dashboard Logic

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadPanel = document.getElementById("upload-panel");
    const processingPanel = document.getElementById("processing-panel");
    const resultsPanel = document.getElementById("results-panel");
    const resetBtn = document.getElementById("reset-btn");
    const loadTestBtn = document.getElementById("load-test-btn");
    const printReportBtn = document.getElementById("print-report-btn");
    
    // Status Lights
    const serverStatusText = document.getElementById("server-status-text");
    const serverStatusDot = document.getElementById("server-status-dot");
    const modelStatusText = document.getElementById("model-status-text");
    const modelStatusDot = document.getElementById("model-status-dot");
    
    // Dashboard Metric Elements
    const previewOriginal = document.getElementById("original-preview");
    const previewAnnotated = document.getElementById("annotated-preview");
    const overallStatusBadge = document.getElementById("overall-status-badge");
    const countBladesEl = document.getElementById("count-blades");
    const countCracksEl = document.getElementById("count-cracks");
    const countBurnsEl = document.getElementById("count-burns");
    const defectTableBody = document.getElementById("defect-table-body");
    
    // Variables
    let lastAnalysisResults = null;
    
    // Initialize Date input to today's date
    const dateInput = document.getElementById("inspection-date");
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;

    // Set server status to online since JavaScript is running
    serverStatusText.textContent = "Server: Online";
    serverStatusDot.className = "pulse-dot green";

    // 1. Drag and Drop Handlers
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    
    dropZone.addEventListener("dragenter", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // 2. Upload file to FastAPI
    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append("file", file);

        // Show processing loader
        uploadPanel.classList.add("hidden");
        processingPanel.classList.remove("hidden");
        resultsPanel.classList.add("hidden");

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server failed to process the image.");
            }

            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            alert(`Error: ${error.message}`);
            resetDashboard();
        }
    }

    // 3. Display Results in Dashboard
    function displayResults(data) {
        lastAnalysisResults = data;
        
        // Hide loader, show results
        processingPanel.classList.add("hidden");
        resultsPanel.classList.remove("hidden");

        // Set Images
        previewOriginal.src = data.original_url;
        previewAnnotated.src = data.result_url;

        // Set metrics counts
        countBladesEl.textContent = data.total_blades;
        countCracksEl.textContent = data.cracks_count;
        countBurnsEl.textContent = data.burns_count;

        // Update model status indicator based on mode
        if (data.is_mock) {
            modelStatusText.textContent = "Model: Demo Mode (Mock)";
            modelStatusDot.className = "pulse-dot orange";
        } else {
            modelStatusText.textContent = "Model: YOLO11 Seg (Loaded)";
            modelStatusDot.className = "pulse-dot green";
        }

        // Determine Overall Status
        let overallStatus = "Passed";
        let hasHighSeverity = false;
        let hasMediumSeverity = false;

        data.blades.forEach(b => {
            if (b.status === "Failed") hasHighSeverity = true;
            if (b.status === "Action Required") hasMediumSeverity = true;
        });

        if (hasHighSeverity) {
            overallStatus = "Failed";
            overallStatusBadge.textContent = "FAILED";
            overallStatusBadge.className = "status-badge failed";
        } else if (hasMediumSeverity) {
            overallStatus = "Action Required";
            overallStatusBadge.textContent = "ACTION REQUIRED";
            overallStatusBadge.className = "status-badge action";
        } else {
            overallStatus = "Passed";
            overallStatusBadge.textContent = "PASSED";
            overallStatusBadge.className = "status-badge passed";
        }

        // Populate Table
        defectTableBody.innerHTML = "";
        
        if (data.blades.length === 0) {
            defectTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--color-muted);">No turbine blades detected.</td></tr>`;
            return;
        }

        let defectRowsAdded = false;

        data.blades.forEach(blade => {
            if (blade.defects.length === 0) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>Blade #${blade.blade_id}</td>
                    <td style="color: var(--color-accent-green);">No defects</td>
                    <td>0 px</td>
                    <td><span class="severity-pill low">None</span></td>
                `;
                defectTableBody.appendChild(row);
                defectRowsAdded = true;
            } else {
                blade.defects.forEach(defect => {
                    const row = document.createElement("tr");
                    const sevClass = defect.severity.toLowerCase();
                    row.innerHTML = `
                        <td>Blade #${blade.blade_id}</td>
                        <td style="text-transform: capitalize; font-weight: 500;">${defect.type}</td>
                        <td>${defect.percent_compromised}% (${defect.area_pixels} px)</td>
                        <td><span class="severity-pill ${sevClass}">${defect.severity}</span></td>
                    `;
                    defectTableBody.appendChild(row);
                    defectRowsAdded = true;
                });
            }
        });

        // Add unassociated defects if any
        if (data.unassociated_defects && data.unassociated_defects.length > 0) {
            data.unassociated_defects.forEach(defect => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td style="color: var(--color-muted);">Unassociated</td>
                    <td style="text-transform: capitalize; font-weight: 500; color: var(--color-muted);">${defect.type}</td>
                    <td>${defect.area_pixels} px</td>
                    <td><span class="severity-pill medium">Medium</span></td>
                `;
                defectTableBody.appendChild(row);
                defectRowsAdded = true;
            });
        }
    }

    // 4. Reset / Clear Upload
    function resetDashboard() {
        fileInput.value = "";
        lastAnalysisResults = null;
        uploadPanel.classList.remove("hidden");
        processingPanel.classList.add("hidden");
        resultsPanel.classList.add("hidden");
    }

    resetBtn.addEventListener("click", resetDashboard);

    // 5. Load Demo/Mock Borescope Image
    loadTestBtn.addEventListener("click", async () => {
        uploadPanel.classList.add("hidden");
        processingPanel.classList.remove("hidden");
        resultsPanel.classList.add("hidden");
        
        try {
            // Fetch the mock test image we generated at root
            const response = await fetch("/static/borescope_test.jpg");
            if (!response.ok) {
                throw new Error("Demo image 'borescope_test.jpg' not found on the server. Please run setup first.");
            }
            
            const blob = await response.blob();
            const file = new File([blob], "borescope_test.jpg", { type: "image/jpeg" });
            handleFileUpload(file);
            
        } catch (error) {
            alert(`Demo Error: ${error.message}`);
            resetDashboard();
        }
    });

    // 6. Generate Print Report Layout
    printReportBtn.addEventListener("click", () => {
        if (!lastAnalysisResults) return;

        // Get Input Metadata
        const engineId = document.getElementById("engine-id").value || "N/A";
        const bladeRow = document.getElementById("blade-row").value || "N/A";
        const inspector = document.getElementById("inspector").value || "N/A";
        const dateVal = document.getElementById("inspection-date").value || today;
        const notes = document.getElementById("notes").value || "No notes entered.";

        // Populate Print Template Text Fields
        document.getElementById("p-engine-id").textContent = engineId;
        document.getElementById("p-blade-row").textContent = bladeRow;
        document.getElementById("p-inspector").textContent = inspector;
        document.getElementById("p-date").textContent = dateVal;
        document.getElementById("p-notes").textContent = notes;
        
        // Report ID based on timestamp
        const reportId = "REP-" + Date.now().toString().slice(-6);
        document.getElementById("p-report-id").textContent = reportId;

        // Set status and images
        const statusText = overallStatusBadge.textContent;
        document.getElementById("p-status").textContent = statusText;
        
        const printOrig = document.getElementById("print-orig-img");
        const printAnn = document.getElementById("print-ann-img");
        printOrig.src = lastAnalysisResults.original_url;
        printAnn.src = lastAnalysisResults.result_url;

        // Populate Print Table Body
        const printTableBody = document.getElementById("print-table-body");
        printTableBody.innerHTML = "";

        lastAnalysisResults.blades.forEach(blade => {
            const tr = document.createElement("tr");
            let defectSummaryText = "Clear (No defects detected)";
            
            if (blade.defects.length > 0) {
                defectSummaryText = blade.defects.map(d => 
                    `${d.type.toUpperCase()} (${d.severity} Severity): compromised ${d.percent_compromised}% of area`
                ).join("<br>");
            }

            tr.innerHTML = `
                <td>Blade #${blade.blade_id}</td>
                <td><strong>${blade.status}</strong></td>
                <td>${blade.defect_count}</td>
                <td>${defectSummaryText}</td>
            `;
            printTableBody.appendChild(tr);
        });

        // Trigger native printer dialog
        // This will print ONLY the #print-report-template card due to print-specific CSS
        window.print();
    });
    
    // Poll server model load state once on UI launch
    setTimeout(async () => {
        try {
            // Standard fetch call to see if model is loaded on server
            const res = await fetch("/");
            if (res.ok) {
                // If model status doesn't change, we will retrieve it in the first upload response.
                modelStatusText.textContent = "Model: Ready (YOLO11)";
                modelStatusDot.className = "pulse-dot green";
            }
        } catch (e) {
            console.log("Failed to query initial model state:", e);
        }
    }, 1500);
});
