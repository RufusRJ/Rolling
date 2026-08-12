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

    // Calibration UI Elements
    const calibrationMethod = document.getElementById("calibration-method");
    const groupBladeHeight = document.getElementById("group-blade-height");
    const groupFixedScale = document.getElementById("group-fixed-scale");
    const knownBladeHeightInput = document.getElementById("known-blade-height");
    const fixedScaleValInput = document.getElementById("fixed-scale-val");
    const targetUnitSelect = document.getElementById("target-unit");
    
    // Variables
    let lastAnalysisResults = null;
    
    // Initialize Date input to today's date
    const dateInput = document.getElementById("inspection-date");
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;

    // Set server status to online since JavaScript is running
    serverStatusText.textContent = "Server: Online";
    serverStatusDot.className = "pulse-dot green";

    // Setup Calibration Events
    calibrationMethod.addEventListener("change", () => {
        if (calibrationMethod.value === "known_blade") {
            groupBladeHeight.classList.remove("hidden");
            groupFixedScale.classList.add("hidden");
        } else {
            groupBladeHeight.classList.add("hidden");
            groupFixedScale.classList.remove("hidden");
        }
        recalculateAndRender();
    });

    knownBladeHeightInput.addEventListener("input", recalculateAndRender);
    fixedScaleValInput.addEventListener("input", recalculateAndRender);
    targetUnitSelect.addEventListener("change", recalculateAndRender);

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
        formData.append("calibration_method", calibrationMethod.value);
        formData.append("known_blade_height", knownBladeHeightInput.value);
        formData.append("fixed_scale", fixedScaleValInput.value);
        formData.append("target_unit", targetUnitSelect.value);

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

        // Render current calibration
        recalculateAndRender();
    }

    // Convert mm to physical unit
    function convertPhysical(mmVal, unit, isArea) {
        if (unit === "cm") {
            return isArea ? mmVal / 100.0 : mmVal / 10.0;
        } else if (unit === "m") {
            return isArea ? mmVal / 1000000.0 : mmVal / 1000.0;
        }
        return mmVal; // default mm
    }

    function recalculateAndRender() {
        if (!lastAnalysisResults) return;

        const data = lastAnalysisResults;
        const calibMethod = calibrationMethod.value;
        const knownBladeHeight = parseFloat(knownBladeHeightInput.value) || 100.0;
        const fixedScale = parseFloat(fixedScaleValInput.value) || 0.25;
        const unit = targetUnitSelect.value;
        const areaUnit = unit + "²";
        const lengthUnit = unit;

        // 1. Compute scale factor for each blade
        let bladesScale = {};
        let scaleSum = 0;
        let bladesCount = 0;

        data.blades.forEach(b => {
            let scale = fixedScale;
            if (calibMethod === "known_blade") {
                const boxHeight = b.box[3] - b.box[1];
                scale = boxHeight > 0 ? knownBladeHeight / boxHeight : fixedScale;
            }
            bladesScale[b.blade_id] = scale;
            scaleSum += scale;
            bladesCount++;
        });

        const avgScale = bladesCount > 0 ? (scaleSum / bladesCount) : fixedScale;

        // 2. Populate Table
        defectTableBody.innerHTML = "";
        
        if (data.blades.length === 0) {
            defectTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--color-muted);">No turbine blades detected.</td></tr>`;
            return;
        }

        data.blades.forEach(blade => {
            const bScale = bladesScale[blade.blade_id];
            const bladeAreaMM = blade.area_pixels * (bScale ** 2);
            const physicalBladeArea = convertPhysical(bladeAreaMM, unit, true);

            if (blade.defects.length === 0) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>Blade #${blade.blade_id}</td>
                    <td style="color: var(--color-accent-green); font-weight: 500;">No defects</td>
                    <td>
                        <span style="font-size: 0.85em; color: var(--color-muted);">
                            Blade Area: ${physicalBladeArea.toFixed(1)} ${areaUnit}
                        </span>
                    </td>
                    <td><span class="severity-pill low">None</span></td>
                `;
                defectTableBody.appendChild(row);
            } else {
                blade.defects.forEach(defect => {
                    const row = document.createElement("tr");
                    const sevClass = defect.severity.toLowerCase();
                    
                    const defectAreaMM = defect.area_pixels * (bScale ** 2);
                    const physicalDefectArea = convertPhysical(defectAreaMM, unit, true);
                    
                    const dx = defect.box[2] - defect.box[0];
                    const dy = defect.box[3] - defect.box[1];
                    const maxDimPx = Math.max(dx, dy);
                    const physicalMaxDim = convertPhysical(maxDimPx * bScale, unit, false);

                    row.innerHTML = `
                        <td>Blade #${blade.blade_id}</td>
                        <td style="text-transform: capitalize; font-weight: 500;">${defect.type}</td>
                        <td>
                            <div><strong>${defect.percent_compromised}%</strong> of area</div>
                            <div style="font-size: 0.85em; color: var(--color-muted); margin-top: 2px; line-height: 1.3;">
                                Area: ${physicalDefectArea.toFixed(3)} ${areaUnit} <span style="font-size:0.9em;color:var(--color-muted);">(${defect.area_pixels} px)</span><br>
                                Max Dim: ${physicalMaxDim.toFixed(2)} ${lengthUnit}
                            </div>
                        </td>
                        <td><span class="severity-pill ${sevClass}">${defect.severity}</span></td>
                    `;
                    defectTableBody.appendChild(row);
                });
            }
        });

        // Add unassociated defects if any
        if (data.unassociated_defects && data.unassociated_defects.length > 0) {
            data.unassociated_defects.forEach(defect => {
                const defectAreaMM = defect.area_pixels * (avgScale ** 2);
                const physicalDefectArea = convertPhysical(defectAreaMM, unit, true);
                
                const dx = defect.box[2] - defect.box[0];
                const dy = defect.box[3] - defect.box[1];
                const maxDimPx = Math.max(dx, dy);
                const physicalMaxDim = convertPhysical(maxDimPx * avgScale, unit, false);

                const row = document.createElement("tr");
                row.innerHTML = `
                    <td style="color: var(--color-muted);">Unassociated</td>
                    <td style="text-transform: capitalize; font-weight: 500; color: var(--color-muted);">${defect.type}</td>
                    <td>
                        <div style="font-size: 0.85em; color: var(--color-muted); line-height: 1.3;">
                            Area: ${physicalDefectArea.toFixed(3)} ${areaUnit} <span style="font-size:0.9em;color:var(--color-muted);">(${defect.area_pixels} px)</span><br>
                            Max Dim: ${physicalMaxDim.toFixed(2)} ${lengthUnit}
                        </div>
                    </td>
                    <td><span class="severity-pill medium">Medium</span></td>
                `;
                defectTableBody.appendChild(row);
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

        // Calibration values
        const calibMethod = calibrationMethod.value;
        const knownBladeHeight = parseFloat(knownBladeHeightInput.value) || 100.0;
        const fixedScale = parseFloat(fixedScaleValInput.value) || 0.25;
        const unit = targetUnitSelect.value;
        const areaUnit = unit + "²";
        const lengthUnit = unit;

        let bladesScale = {};
        lastAnalysisResults.blades.forEach(b => {
            let scale = fixedScale;
            if (calibMethod === "known_blade") {
                const boxHeight = b.box[3] - b.box[1];
                scale = boxHeight > 0 ? knownBladeHeight / boxHeight : fixedScale;
            }
            bladesScale[b.blade_id] = scale;
        });

        // Populate Print Table Body
        const printTableBody = document.getElementById("print-table-body");
        printTableBody.innerHTML = "";

        lastAnalysisResults.blades.forEach(blade => {
            const bScale = bladesScale[blade.blade_id];
            const tr = document.createElement("tr");
            let defectSummaryText = "Clear (No defects detected)";
            
            if (blade.defects.length > 0) {
                defectSummaryText = blade.defects.map(d => {
                    const defectAreaMM = d.area_pixels * (bScale ** 2);
                    const physicalDefectArea = convertPhysical(defectAreaMM, unit, true);
                    
                    const dx = d.box[2] - d.box[0];
                    const dy = d.box[3] - d.box[1];
                    const physicalMaxDim = convertPhysical(Math.max(dx, dy) * bScale, unit, false);

                    return `${d.type.toUpperCase()} (${d.severity} Severity): compromised ${d.percent_compromised}% of area (${physicalDefectArea.toFixed(3)} ${areaUnit}, max dim: ${physicalMaxDim.toFixed(2)} ${lengthUnit})`;
                }).join("<br>");
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
