// RAGInGoa Client Application Logic

// API Configuration defaults
const API_BASE_URL = (window.location.protocol === 'file:' || !window.location.host.includes(':8000'))
    ? 'http://127.0.0.1:8000'
    : window.location.origin;

// State Variables
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioStream = null;

// Typewriter Animation state
let typingInterval = null;

// Initial Setup on DOM Content Loaded
document.addEventListener("DOMContentLoaded", () => {
    loadConfig();
    setupTabs();
    setupRecorder();
    setupTextInput();
    setupSandbox();
    setupAnalytics();
    setupScrollEffects();
});

// 1. Fetch active backend configuration settings
async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/config`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById("config-lang").textContent = `Lang: ${data.dataset_language}`;
            document.getElementById("config-chunker").textContent = `Chunker: ${data.chunking_strategy}`;
            document.getElementById("config-model").textContent = `Embedder: ${data.embedding_model.split('/').pop()}`;
        }
    } catch (e) {
        console.error("Failed to load config:", e);
    }
}

// 2. Navigation Tab System
function setupTabs() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            // Remove active states
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            
            // Set current tab active
            tab.classList.add("active");
            const contentId = tab.getAttribute("data-tab");
            document.getElementById(contentId).classList.add("active");
            
            // Fetch fresh stats if navigating to Analytics Tab
            if (contentId === "latency-analytics-tab") {
                loadAnalytics();
            }
        });
    });
}

// 2b. Scroll Effects & Nav Interceptors
function setupScrollEffects() {
    // Header scroll shrink handler
    window.addEventListener("scroll", () => {
        const header = document.getElementById("sticky-header");
        if (header) {
            if (window.scrollY > 50) {
                header.classList.add("shrink");
            } else {
                header.classList.remove("shrink");
            }
        }
    });

    // Intercept header link clicks to switch tabs automatically
    const navLinks = document.querySelectorAll(".nav-link");
    navLinks.forEach(link => {
        link.addEventListener("click", () => {
            const href = link.getAttribute("href");
            let targetTabId = "";
            
            if (href === "#rag-lab-anchor" || href === "#pipeline-flow-anchor") {
                targetTabId = "voice-rag-tab";
            } else if (href === "#chunking-sandbox-anchor") {
                targetTabId = "chunking-sandbox-tab";
            } else if (href === "#latency-analytics-anchor") {
                targetTabId = "latency-analytics-tab";
            }
            
            if (targetTabId) {
                const tabBtn = document.querySelector(`.tab-btn[data-tab="${targetTabId}"]`);
                if (tabBtn && !tabBtn.classList.contains("active")) {
                    tabBtn.click();
                }
            }
        });
    });
}

let recognition = null;
let activeLang = "hi-IN"; // Default recognition language BCP-47 for Hindi

// 3. Audio Recording and SpeechRecognition Wrapper
function setupRecorder() {
    const micBtn = document.getElementById("mic-btn");
    const stateText = document.getElementById("recording-state-text");
    const sttContainer = document.getElementById("stt-container");
    const sttTranscript = document.getElementById("stt-transcript");
    const queryInput = document.getElementById("query-input");
    const liveInd = document.getElementById("live-indicator");
    
    // Toggle language by clicking the state text span
    if (stateText) {
        stateText.style.cursor = "pointer";
        stateText.title = "Click to toggle between Hindi and English voice input";
        stateText.addEventListener("click", () => {
            if (isRecording) return; // Prevent toggling while recording
            if (activeLang === "hi-IN") {
                activeLang = "en-IN";
                stateText.textContent = "English Voice Input";
            } else {
                activeLang = "hi-IN";
                stateText.textContent = "Hindi Voice Input";
            }
        });
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn("SpeechRecognition not supported in this browser.");
        if (stateText) {
            stateText.textContent = "Voice Input Unsupported";
        }
        micBtn.addEventListener("click", () => {
            alert("Speech recognition is not supported in this browser. Please use Google Chrome or Safari.");
        });
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart = () => {
        isRecording = true;
        micBtn.classList.add("recording");
        stateText.textContent = "Listening... Tap to Stop";
        if (liveInd) liveInd.classList.remove("hidden");
        if (sttContainer) {
            sttContainer.classList.remove("hidden");
            sttTranscript.textContent = "Listening...";
        }
    };

    recognition.onresult = (event) => {
        let interimTranscript = "";
        let finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        const currentText = finalTranscript || interimTranscript;
        if (sttTranscript && currentText) {
            sttTranscript.textContent = currentText;
        }
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        if (event.error === "not-allowed") {
            alert("Microphone permission denied. Please allow microphone access in your browser settings.");
        } else if (event.error === "network") {
            alert("Network error occurred during speech recognition. Please check your internet connection.");
        } else if (event.error !== "aborted") {
            alert("Speech recognition error: " + event.error);
        }
        cleanupRecordingState();
    };

    recognition.onend = () => {
        const transcriptText = sttTranscript.textContent;
        if (transcriptText && transcriptText !== "Listening..." && transcriptText !== "Transcribing speech...") {
            if (queryInput) {
                queryInput.value = transcriptText;
            }
        }
        cleanupRecordingState();
    };

    function cleanupRecordingState() {
        isRecording = false;
        micBtn.classList.remove("recording");
        if (liveInd) liveInd.classList.add("hidden");
        if (stateText) {
            stateText.textContent = activeLang === "hi-IN" ? "Hindi Voice Input" : "English Voice Input";
        }
    }

    micBtn.addEventListener("click", () => {
        if (!isRecording) {
            recognition.lang = activeLang;
            try {
                recognition.start();
            } catch (e) {
                console.error("Error starting speech recognition:", e);
            }
        } else {
            recognition.stop();
        }
    });
}

// 4. Text Input submission
function setupTextInput() {
    const queryInput = document.getElementById("query-input");
    const submitBtn = document.getElementById("submit-btn");
    
    const triggerTextQuery = async () => {
        const query = queryInput.value.trim();
        if (!query) return;
        queryInput.value = "";
        await sendTextQuery(query);
    };
    
    submitBtn.addEventListener("click", triggerTextQuery);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            triggerTextQuery();
        }
    });
}

// 5. Send Text Query to API
async function sendTextQuery(query) {
    // Hide transcription area for plain text query
    document.getElementById("stt-container").classList.add("hidden");
    
    const formData = new FormData();
    formData.append("query", query);
    formData.append("language", "hi-IN"); // BCP-47 for Hindi default
    
    await executeRAGPipeline(formData);
}

// 6. Send Recorded Audio Blob to API
async function sendAudioQuery(audioBlob) {
    const formData = new FormData();
    // We name the file audio.wav (the backend processes it generically as multi-part file)
    formData.append("file", audioBlob, "audio.wav");
    formData.append("language", "hi-IN");
    
    await executeRAGPipeline(formData);
}

// 7. Core RAG Pipeline trigger and response formatter
async function executeRAGPipeline(formData) {
    // Show loading state
    const answerEl = document.getElementById("final-answer");
    const badgeEl = document.getElementById("guardrail-badge");
    const e2eEl = document.getElementById("latency-e2e");
    const retEl = document.getElementById("latency-ret");
    const passagesList = document.getElementById("retrieved-passages-list");
    const resultsCount = document.getElementById("results-count");
    const resultsCountStrip = document.getElementById("results-count-strip");
    const resultsCountStats = document.getElementById("results-count-stats");
    const answerCard = document.querySelector(".answer-glass-card");
    
    // Clear and set loading values
    if (answerCard) answerCard.className = "answer-glass-card";
    badgeEl.textContent = "VERIFYING...";
    badgeEl.className = "badge credential-badge";
    e2eEl.textContent = "---";
    retEl.textContent = "---";
    if (resultsCountStrip) resultsCountStrip.textContent = "---";
    if (resultsCountStats) resultsCountStats.textContent = "---";
    
    passagesList.innerHTML = '<div class="empty-passages">Retrieving contexts from database...</div>';
    resultsCount.textContent = "0 DOCUMENTS";
    
    // Start timeline progress animation
    const timeline = document.getElementById("pipeline-visual-connection");
    if (timeline) {
        timeline.className = "pipeline-progress-timeline active-query";
    }
    
    // Animate typing text indicating progress
    animateText("Retrieving relevant passages and running guardrails...");
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/rag`, {
            method: "POST",
            body: formData
        });
        
        // Reset mic label in case it was transcribing
        document.getElementById("recording-state-text").textContent = "Hindi Voice Input";
        const liveInd = document.getElementById("live-indicator");
        if (liveInd) liveInd.classList.add("hidden");
        
        if (response.ok) {
            const data = await response.json();
            
            // If Speech-to-Text was performed, display transcription
            if (data.stt_transcript) {
                document.getElementById("stt-container").classList.remove("hidden");
                document.getElementById("stt-transcript").textContent = data.stt_transcript;
            }
            
            // Format E2E Latencies
            e2eEl.textContent = data.overall_latency_ms.toFixed(2);
            retEl.textContent = data.latency_breakdown.retrieval_ms.toFixed(2);
            
            // Apply Guardrail Badge styling
            const guards = data.guardrail_status;
            if (data.generation_mode === "ERROR") {
                badgeEl.textContent = "API / EXECUTION ERROR";
                badgeEl.className = "badge credential-badge badge-error";
                if (answerCard) answerCard.classList.add("grounding-fail");
            } else if (!guards.safety_passed) {
                badgeEl.textContent = "SAFETY FLAGGED";
                badgeEl.className = "badge credential-badge badge-error";
                if (answerCard) answerCard.classList.add("safety-fail");
            } else if (!guards.topic_passed) {
                badgeEl.textContent = "OFF-TOPIC BLOCK";
                badgeEl.className = "badge credential-badge badge-warning";
                if (answerCard) answerCard.classList.add("grounding-fail");
            } else if (!guards.context_sufficient) {
                badgeEl.textContent = "INSUFFICIENT CONTEXT";
                badgeEl.className = "badge credential-badge badge-warning";
                if (answerCard) answerCard.classList.add("grounding-fail");
            } else if (!guards.grounding_passed) {
                badgeEl.textContent = "HALLUCINATION DETECTED";
                badgeEl.className = "badge credential-badge badge-error";
                if (answerCard) answerCard.classList.add("grounding-fail");
            } else {
                badgeEl.textContent = "● " + (data.generation_mode || "GROUNDED AI ANSWER");
                badgeEl.className = "badge credential-badge badge-success";
            }
            
            // Step-by-step progress timeline animation sequence
            if (timeline) {
                setTimeout(() => timeline.className = "pipeline-progress-timeline active-query active-retrieve", 150);
                setTimeout(() => timeline.className = "pipeline-progress-timeline active-query active-retrieve active-ground", 300);
                setTimeout(() => timeline.className = "pipeline-progress-timeline active-query active-retrieve active-ground active-answer", 450);
            }
            
            // Typewrite the final answer text
            animateText(data.answer);
            
            // Render context passages
            const contexts = data.contexts || [];
            resultsCount.textContent = `${contexts.length}`;
            if (resultsCountStrip) resultsCountStrip.textContent = contexts.length;
            if (resultsCountStats) resultsCountStats.textContent = contexts.length;
            
            if (contexts.length === 0) {
                passagesList.innerHTML = '<div class="empty-passages">No context passages were retrieved for this query.</div>';
            } else {
                passagesList.innerHTML = "";
                contexts.forEach((ctx, idx) => {
                    const meta = ctx.metadata || {};
                    const score = ctx.score || 0.0;
                    const scorePercent = (score * 100).toFixed(2);
                    
                    const item = document.createElement("div");
                    item.className = "passage-item";
                    
                    let html = `
                        <div class="passage-meta">
                            <span>DOCUMENT [0${idx+1}] – ID: ${meta.query_id || 'N/A'}</span>
                            <span class="passage-score">Similarity: ${scorePercent}%</span>
                        </div>
                        <div class="evidence-score-visual-bar">
                            <span>SCORE</span>
                            <div class="score-progress-track">
                                <div class="score-progress-fill" style="width: ${scorePercent}%"></div>
                            </div>
                        </div>
                        <p class="passage-text">${ctx.text}</p>
                    `;
                    
                    if (meta.english_text) {
                        html += `
                            <details class="passage-english">
                                <summary style="cursor: pointer; color: var(--cyan); font-family: var(--font-mono); font-size: 0.72rem; letter-spacing:1px;">VIEW ORIGINAL ENGLISH</summary>
                                <p style="padding-top: 0.35rem; font-style: italic; color: var(--cream-body);">${meta.english_text}</p>
                            </details>
                        `;
                    }
                    
                    item.innerHTML = html;
                    passagesList.appendChild(item);
                });
            }
            
            // Reload analytics background stats
            loadAnalytics();
            
        } else {
            const err = await response.json();
            badgeEl.textContent = "ERROR";
            badgeEl.className = "badge credential-badge badge-error";
            animateText(`Execution failed: ${err.detail || "Server error"}`);
            passagesList.innerHTML = '<div class="empty-passages">Search execution failed.</div>';
            if (timeline) timeline.className = "pipeline-progress-timeline";
        }
    } catch (e) {
        document.getElementById("recording-state-text").textContent = "Hindi Voice Input";
        const liveInd = document.getElementById("live-indicator");
        if (liveInd) liveInd.classList.add("hidden");
        badgeEl.textContent = "API ERROR";
        badgeEl.className = "badge credential-badge badge-error";
        
        let errorMessage = `API CONNECTION ERROR\n\nUnable to reach RAGInGoa backend.\n\nBackend URL: ${API_BASE_URL}\n\nCheck that the FastAPI server is running.`;
        if (e.message && e.message !== "Failed to fetch") {
            errorMessage += `\n\nDetails: ${e.message}`;
        }
        
        animateText(errorMessage);
        passagesList.innerHTML = '<div class="empty-passages">Network connection error.</div>';
        if (timeline) timeline.className = "pipeline-progress-timeline";
    }
}

// 8. Typewriter text animation helper
function animateText(text) {
    const answerEl = document.getElementById("final-answer");
    const cursor = document.getElementById("typing-cursor");
    
    // Stop any current running typing interval
    clearInterval(typingInterval);
    cursor.classList.remove("hidden");
    
    answerEl.textContent = "";
    let i = 0;
    
    typingInterval = setInterval(() => {
        if (i < text.length) {
            answerEl.textContent += text.charAt(i);
            i++;
        } else {
            clearInterval(typingInterval);
            cursor.classList.add("hidden");
        }
    }, 8); // Fast typing speed
}

// 9. Chunking Compare Sandbox controls
function setupSandbox() {
    const textSlider = document.getElementById("param-chunk-size");
    const overlapSlider = document.getElementById("param-chunk-overlap");
    const simSlider = document.getElementById("param-similarity-threshold");
    
    const textVal = document.getElementById("val-chunk-size");
    const overlapVal = document.getElementById("val-chunk-overlap");
    const simVal = document.getElementById("val-similarity");
    
    // Add slide listeners
    textSlider.addEventListener("input", () => textVal.textContent = textSlider.value);
    overlapSlider.addEventListener("input", () => overlapVal.textContent = overlapSlider.value);
    simSlider.addEventListener("input", () => simVal.textContent = simSlider.value);
    
    const sampleText = (
        "निगम (Corporation) राज्य की दृष्टि में एक कृत्रिम व्यक्ति है जिसे कानून द्वारा अलग अस्तित्व दिया गया है। " +
        "यह अपने सदस्यों से स्वतंत्र रहकर संपत्ति खरीद सकता है, अनुबंध कर सकता है और कानूनी विवादों में शामिल हो सकता है। " +
        "आज के आधुनिक उद्योग जगत में निगमों की भूमिका काफी महत्वपूर्ण है। लगभग 50 देशों के 2,100 से अधिक प्रमाणित निगम एक साझा लक्ष्य के साथ काम कर रहे हैं। " +
        "एक निगम अपने शेयरधारकों के स्वामित्व में होता है जो इसके मुनाफे और घाटे को साझा करते हैं। इसके विपरीत, एक सामान्य साझेदारी संगठन कानून में एकल इकाई नहीं होता।"
    );
    document.getElementById("sandbox-text").value = sampleText;
    
    const compareBtn = document.getElementById("compare-chunks-btn");
    compareBtn.addEventListener("click", async () => {
        const text = document.getElementById("sandbox-text").value.trim();
        if (!text) {
            alert("Please input some text to partition.");
            return;
        }
        
        compareBtn.textContent = "Processing splits...";
        compareBtn.disabled = true;
        
        const formData = new FormData();
        formData.append("text", text);
        formData.append("chunk_size", textSlider.value);
        formData.append("chunk_overlap", overlapSlider.value);
        formData.append("similarity_threshold", simSlider.value);
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/chunking/compare`, {
                method: "POST",
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                
                renderChunkColumn("col-fixed-list", "count-fixed", data.fixed_size);
                renderChunkColumn("col-semantic-list", "count-semantic", data.semantic);
                renderChunkColumn("col-metadata-list", "count-metadata", data.metadata);
            } else {
                alert("Chunking compilation failed.");
            }
        } catch (e) {
            console.error(e);
            alert("Failed to connect to chunking comparator endpoint.");
        } finally {
            compareBtn.textContent = "Execute Comparison";
            compareBtn.disabled = false;
        }
    });
}

function renderChunkColumn(colListId, countId, chunks) {
    const listEl = document.getElementById(colListId);
    const countEl = document.getElementById(countId);
    
    countEl.textContent = `${chunks.length} Chunks`;
    listEl.innerHTML = "";
    
    if (chunks.length === 0) {
        listEl.innerHTML = '<div class="empty-list-placeholder">No chunks generated.</div>';
        return;
    }
    
    chunks.forEach(c => {
        const div = document.createElement("div");
        div.className = "chunk-item";
        div.innerHTML = `
            <div class="chunk-item-meta">
                <span>CHUNK [${c.index + 1}]</span>
                <span>Length: ${c.length} chars</span>
            </div>
            <p>${c.text}</p>
        `;
        listEl.appendChild(div);
    });
}

// 10. Latency Dashboard and Metrics loader
function setupAnalytics() {
    document.getElementById("refresh-analytics-btn").addEventListener("click", loadAnalytics);
}

async function loadAnalytics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/analytics`);
        if (response.ok) {
            const data = await response.json();
            
            // Update Dashboard cards
            document.getElementById("metric-p50").textContent = `${data.p50.toFixed(2)} ms`;
            document.getElementById("metric-p70").textContent = `${data.p70.toFixed(2)} ms`;
            document.getElementById("metric-p100").textContent = `${data.p100.toFixed(2)} ms`;
            document.getElementById("metric-avg").textContent = `${data.avg.toFixed(2)} ms`;
            document.getElementById("total-queries-badge").textContent = `${data.count} Queries Total`;
            
            // Render bars relative to the maximum percentile value or 1000ms threshold
            const maxVal = Math.max(data.p50, 100.0); // normalize scale
            
            const updateBar = (barId, labelId, value) => {
                const bar = document.getElementById(barId);
                const label = document.getElementById(labelId);
                label.textContent = `${value.toFixed(2)} ms`;
                const percentage = Math.min((value / maxVal) * 100.0, 100.0);
                bar.style.width = `${percentage}%`;
            };
            
            updateBar("time-stt-bar", "time-stt-label", data.stt_p50);
            updateBar("time-ret-bar", "time-ret-label", data.retrieval_p50);
            updateBar("time-gen-bar", "time-gen-label", data.generation_p50);
            updateBar("time-guard-bar", "time-guard-label", data.guardrails_p50);
            
            // Render logs table
            const tbody = document.getElementById("logs-tbody");
            tbody.innerHTML = "";
            
            const logs = data.details || [];
            if (logs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="empty-table-placeholder">No pipeline executions logged yet.</td></tr>';
            } else {
                // Reverse log list so newest show on top
                const reversedLogs = [...logs].reverse();
                reversedLogs.forEach(run => {
                    const tr = document.createElement("tr");
                    
                    let statusHtml = '<span class="badge badge-success" style="font-size:0.65rem;">PASSED</span>';
                    if (!run.safety || !run.grounding) {
                        statusHtml = '<span class="badge badge-error" style="font-size:0.65rem;">FLAGGED</span>';
                    }
                    
                    tr.innerHTML = `
                        <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${run.query}</td>
                        <td style="font-weight: bold; color: var(--sunset-gold);">${run.latency_ms.toFixed(1)} ms</td>
                        <td>${run.stt_ms.toFixed(1)} ms</td>
                        <td>${run.retrieval_ms.toFixed(1)} ms</td>
                        <td>${run.generation_ms.toFixed(1)} ms</td>
                        <td>${run.guardrails_ms.toFixed(1)} ms</td>
                        <td>${statusHtml}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        }
    } catch (e) {
        console.error("Failed to load analytics data:", e);
    }
}
