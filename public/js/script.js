// Set this to your backend's public URL
const BACKEND_URL = 'https://snoopy1.usc.edu:8443'

console.log('SCRIPT EXECUTION STARTED. Backend URL:', BACKEND_URL);
console.log('Current hostname:', window.location.hostname);

// Global state
let currentSession = null;
    let currentPairIndex = 0;
let maxSegments = 50000;
    let preferences = [];
    let sessionId = new Date().getTime().toString();
let isLoading = false;
let hasError = false;
let currentDataset = 'assembly-v2';
let datasetInfo = {
    total_pairs: 0,
    labeled_pairs: 0,
    unlabeled_pairs: 0,
    total_segments: 0
};

// Function to get random pair index
function getRandomPairIndex() {
    if (maxSegments <= 0) {
        return 0;
    }
    return Math.floor(Math.random() * maxSegments);
}

// Global state for active learning
let isActiveSession = false;
let maxIterations = 10;
let currentIteration = 0;
let currentAcquisition = 'disagreement';
let activeSessionId = null;

// Global DOM elements
let elements = {};

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', initialize);

// Function to load a video and return a promise
function loadVideo(videoElement, url) {
    return new Promise((resolve, reject) => {
        if (!videoElement) {
            reject(new Error('Video element not found'));
            return;
        }

        // Set up event listeners
        const handleLoad = () => {
            console.log(`Video loaded successfully: ${url}`);
            cleanup();
            resolve();
        };

        const handleError = (error) => {
            console.error(`Error loading video ${url}:`, error);
            cleanup();
            reject(new Error(`Failed to load video: ${url}`));
        };

        const cleanup = () => {
            videoElement.removeEventListener('loadeddata', handleLoad);
            videoElement.removeEventListener('error', handleError);
        };

        // Add event listeners
        videoElement.addEventListener('loadeddata', handleLoad);
        videoElement.addEventListener('error', handleError);

        // Set video properties
        videoElement.loop = true;
        videoElement.muted = true;
        videoElement.src = url;
        videoElement.load();
    });
}

// Initialize the application
async function initialize() {
    try {
        console.log('Initializing application...');
        initializeDOMElements();
        setupEventListeners();
        await loadAvailableDatasets();
        await updateDatasetInfo();
        
        // Start with a random pair after dataset info is loaded
        currentPairIndex = getRandomPairIndex();
        console.log(`Starting with random pair index: ${currentPairIndex} (max: ${maxSegments})`);
        loadTrajectoryPair();
        
        console.log('Initialization complete');
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to initialize application: ' + error.message);
    }
}

// Initialize DOM elements
function initializeDOMElements() {
    console.log('Initializing DOM elements...');
    
    // Initialize basic UI elements
    elements.tabButtons = document.querySelectorAll('.tab-button');
    elements.tabContents = document.querySelectorAll('.tab-content');
    elements.trajectoryAVideo = document.getElementById('trajectory-A');
    elements.trajectoryBVideo = document.getElementById('trajectory-B');
    elements.preferAButton = document.getElementById('prefer-A');
    elements.preferEqualButton = document.getElementById('prefer-equal');
    elements.preferBButton = document.getElementById('prefer-B');
    elements.prevPairButton = document.getElementById('prev-pair');
    elements.nextPairButton = document.getElementById('next-pair');
    elements.randomPairButton = document.getElementById('random-pair');
    elements.pairIndexInput = document.getElementById('pair-index');
    elements.goToPairButton = document.getElementById('go-to-pair');
    elements.currentPairIndexDisplay = document.getElementById('current-pair-index');
    elements.datasetSelect = document.getElementById('dataset-select');
    
    // Initialize active learning elements
    elements.startActiveSessionButton = document.getElementById('startActiveSession');
    elements.acquisitionSelect = document.getElementById('acquisition-select');
    elements.maxIterationsSelect = document.getElementById('max-iterations');
    elements.activeProgressBar = document.getElementById('active-progress');
    elements.activeLabeledCount = document.getElementById('active-labeled-count');
    elements.activeUnlabeledCount = document.getElementById('active-unlabeled-count');
    elements.activeTotalCount = document.getElementById('active-total-count');
    elements.activeTrajectoryAVideo = document.getElementById('active-trajectory-A');
    elements.activeTrajectoryBVideo = document.getElementById('active-trajectory-B');
    elements.activePreferAButton = document.getElementById('active-prefer-A');
    elements.activePreferEqualButton = document.getElementById('active-prefer-equal');
    elements.activePreferBButton = document.getElementById('active-prefer-B');
    
    // Initialize similar segments elements
    elements.segmentIndexInput = document.getElementById('segment-index-input');
    elements.kValueInput = document.getElementById('k-value');
    elements.dtwTypeSelect = document.getElementById('dtw-type-select');
    elements.findSimilarButton = document.getElementById('find-similar');
    elements.targetSegmentVideo = document.getElementById('target-segment-video');
    elements.targetSegmentIndex = document.getElementById('target-segment-index');
    elements.targetSegmentReward = document.getElementById('target-segment-reward');
    elements.similarSegmentsContainer = document.getElementById('similar-segments-container');
    elements.dissimilarSegmentsContainer = document.getElementById('dissimilar-segments-container');
    
    // Initialize error handling elements
    elements.errorPopup = document.getElementById('error-popup');
    elements.errorMessage = document.getElementById('error-message');
    
    // Log which elements were found and which weren't
    console.log('DOM Elements initialization status:');
    Object.entries(elements).forEach(([key, value]) => {
        console.log(`${key}: ${value ? 'Found' : 'Not found'}`);
    });
    
    // Set initial active tab
    switchTab('preference-tab');
}

// Set up event listeners
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Tab buttons
    if (elements.tabButtons) {
        elements.tabButtons.forEach(button => {
            if (button) {
                button.addEventListener('click', () => {
                    const tabId = button.getAttribute('data-tab');
                    switchTab(tabId);
                });
            }
        });
    }
    
    // Regular preference collection buttons
    if (elements.preferAButton) {
        elements.preferAButton.addEventListener('click', () => handlePreferA());
    }
    if (elements.preferEqualButton) {
        elements.preferEqualButton.addEventListener('click', () => handlePreferEqual());
    }
    if (elements.preferBButton) {
        elements.preferBButton.addEventListener('click', () => handlePreferB());
    }
    
    // Navigation buttons
    if (elements.prevPairButton) {
        elements.prevPairButton.addEventListener('click', showPreviousPair);
    }
    if (elements.nextPairButton) {
        elements.nextPairButton.addEventListener('click', showNextPair);
    }
    if (elements.randomPairButton) {
        elements.randomPairButton.addEventListener('click', () => {
            currentPairIndex = getRandomPairIndex();
            console.log(`Random pair button clicked, loading pair: ${currentPairIndex}`);
            loadTrajectoryPair();
        });
    }
    if (elements.goToPairButton) {
        elements.goToPairButton.addEventListener('click', goToPair);
    }
    
    // Dataset selector
    if (elements.datasetSelect) {
        elements.datasetSelect.addEventListener('change', async (event) => {
            currentDataset = event.target.value;
            await updateDatasetInfo();
            
            // Load a random pair when dataset changes (after dataset info is updated)
            currentPairIndex = getRandomPairIndex();
            console.log(`Dataset changed to ${currentDataset}, loading random pair: ${currentPairIndex} (max: ${maxSegments})`);
            loadTrajectoryPair();
            
            // Update segment index input for similar segments tab with new dataset bounds
            if (elements.segmentIndexInput && datasetInfo.total_segments > 0) {
                const randomSegmentIndex = Math.floor(Math.random() * datasetInfo.total_segments);
                elements.segmentIndexInput.value = randomSegmentIndex;
                console.log(`Updated segment index for new dataset to: ${randomSegmentIndex} (max: ${datasetInfo.total_segments - 1})`);
                
                // If we're currently on the similar segments tab, automatically load the new random segment
                const currentTab = document.querySelector('.tab-content.active');
                if (currentTab && currentTab.id === 'similar-tab') {
                    findSimilarSegments();
                }
            }
        });
    }
    
    // Active learning controls
    if (elements.startActiveSessionButton) {
        elements.startActiveSessionButton.addEventListener('click', () => {
            console.log('Start Active Session button clicked');
            startNewActiveSession();
        });
    }
    if (elements.acquisitionSelect) {
        elements.acquisitionSelect.addEventListener('change', handleAcquisitionChange);
    }
    if (elements.maxIterationsSelect) {
        elements.maxIterationsSelect.addEventListener('change', (event) => {
            maxIterations = parseInt(event.target.value);
        });
    }
    
    // Active learning preference buttons
    if (elements.activePreferAButton) {
        elements.activePreferAButton.addEventListener('click', () => handleActivePreferA());
    }
    if (elements.activePreferEqualButton) {
        elements.activePreferEqualButton.addEventListener('click', () => handleActivePreferEqual());
    }
    if (elements.activePreferBButton) {
        elements.activePreferBButton.addEventListener('click', () => handleActivePreferB());
    }
    
    // Similar segments controls
    if (elements.findSimilarButton) {
        elements.findSimilarButton.addEventListener('click', findSimilarSegments);
    }
    
    // DTW type selector
    if (elements.dtwTypeSelect) {
        elements.dtwTypeSelect.addEventListener('change', () => {
            // Re-run search when DTW type changes if we have a segment loaded
            if (elements.segmentIndexInput && elements.segmentIndexInput.value) {
                findSimilarSegments();
            }
        });
    }
    
    // Random segment button
    const randomSegmentButton = document.getElementById('random-segment');
    if (randomSegmentButton) {
        randomSegmentButton.addEventListener('click', loadRandomSimilarSegment);
    }
    
    // Note: segment index input will be set after dataset info is loaded
    
    // Error popup buttons
    const errorReloadButton = document.getElementById('error-reload');
    const errorCloseButton = document.getElementById('error-close');
    
    if (errorReloadButton) {
        errorReloadButton.addEventListener('click', () => {
            window.location.reload();
        });
    }
    
    if (errorCloseButton) {
        errorCloseButton.addEventListener('click', () => {
            hasError = false;
            if (elements.errorPopup) {
                elements.errorPopup.classList.add('hidden');
            }
        });
    }
    
    console.log('Event listeners set up successfully');
}

// Regular preference handlers
function handlePreferA() {
    recordPreference('A');
}

function handlePreferEqual() {
    recordPreference('equal');
}

function handlePreferB() {
    recordPreference('B');
}

// Active learning preference handlers
function handleActivePreferA() {
    recordActivePreference('A');
}

function handleActivePreferEqual() {
    recordActivePreference('equal');
}

function handleActivePreferB() {
    recordActivePreference('B');
}

// Load available datasets
async function loadAvailableDatasets() {
    try {
        console.log('Loading available datasets...');
        const response = await safeFetch(`${BACKEND_URL}/api/get-available-datasets`);
        const data = await response.json();
        
        if (elements.datasetSelect) {
            elements.datasetSelect.innerHTML = '';
            data.datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset;
                option.textContent = dataset;
                if (dataset === currentDataset) {
                    option.selected = true;
                }
                elements.datasetSelect.appendChild(option);
            });
        }
        
        console.log(`Loaded ${data.datasets.length} datasets`);
    } catch (error) {
        console.error('Error loading datasets:', error);
        showError('Failed to load datasets: ' + error.message);
    }
}

// Show error message
function showError(message) {
    console.error('Showing error:', message);
    hasError = true;
    
    // Convert Error objects to string
    const errorText = message instanceof Error ? message.message : String(message);
    
    if (elements.errorMessage) {
        elements.errorMessage.textContent = errorText;
    }
    if (elements.errorPopup) {
        elements.errorPopup.classList.remove('hidden');
    }
}

// Show training popup
function showTrainingPopup(message = 'Training reward model...', showSpinner = true, isSuccess = false) {
    console.log('Showing training popup:', message);
    const trainingPopup = document.getElementById('training-popup');
    const trainingStatus = document.getElementById('training-status');
    const trainingContent = document.querySelector('.training-content');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const trainingTitle = document.querySelector('.training-content h3');
    
    if (trainingStatus) {
        trainingStatus.textContent = message;
    }
    
    // Update title based on success state
    if (trainingTitle) {
        trainingTitle.textContent = isSuccess ? 'Training Complete!' : 'Training Reward Model';
    }
    
    // Show or hide spinner based on the type of message
    if (loadingSpinner) {
        if (showSpinner) {
            loadingSpinner.style.display = 'block';
        } else {
            loadingSpinner.style.display = 'none';
        }
    }
    
    // Change styling for success messages
    if (trainingContent) {
        if (isSuccess) {
            trainingContent.style.borderLeft = '5px solid #27ae60';
        } else {
            trainingContent.style.borderLeft = '5px solid #3498db';
        }
    }
    
    if (trainingPopup) {
        trainingPopup.classList.remove('hidden');
    }
}

// Hide training popup
function hideTrainingPopup() {
    console.log('Hiding training popup');
    const trainingPopup = document.getElementById('training-popup');
    if (trainingPopup) {
        trainingPopup.classList.add('hidden');
    }
}

// Update dataset info
async function updateDatasetInfo() {
    try {
        console.log('Updating dataset info for:', currentDataset);
        const response = await safeFetch(`${BACKEND_URL}/api/get-dataset-info?dataset=${currentDataset}`);
        const data = await response.json();
        
        datasetInfo = data;
        
        console.log(`Dataset info updated: ${data.total_pairs} pairs, ${data.total_segments} segments`);
        
        // Update regular preference collection info
        const totalCountElement = document.getElementById('total-count');
        const labeledCountElement = document.getElementById('labeled-count'); 
        const progressPercentElement = document.getElementById('progress-percent');
        
        if (totalCountElement) totalCountElement.textContent = data.total_pairs;
        if (labeledCountElement) labeledCountElement.textContent = data.labeled_pairs;
        if (progressPercentElement) {
            const progress = data.total_pairs > 0 ? Math.round((data.labeled_pairs / data.total_pairs) * 100) : 0;
            progressPercentElement.textContent = progress;
        }
        
        // Update active learning info
        if (elements.activeTotalCount) elements.activeTotalCount.textContent = data.total_pairs;
        if (elements.activeLabeledCount) elements.activeLabeledCount.textContent = data.labeled_pairs;
        if (elements.activeUnlabeledCount) elements.activeUnlabeledCount.textContent = data.unlabeled_pairs;
        
        maxSegments = data.total_pairs;
        
        // Set initial segment index input value with proper bounds
        if (elements.segmentIndexInput && !elements.segmentIndexInput.value && data.total_segments > 0) {
            const randomSegmentIndex = Math.floor(Math.random() * data.total_segments);
            elements.segmentIndexInput.value = randomSegmentIndex;
            console.log(`Set initial segment index to: ${randomSegmentIndex}`);
        }
        
        console.log('Dataset info updated successfully');
    } catch (error) {
        console.error('Error updating dataset info:', error);
        showError('Failed to update dataset info: ' + error.message);
    }
}

// Function to make a fetch request with proper error handling
async function safeFetch(url, options = {}) {
    try {
        console.log(`Fetching: ${url}`);
        const response = await fetch(url, {
            ...options,
            credentials: 'include',
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response;
    } catch (error) {
        console.error(`Fetch error for ${url}:`, error);
        throw error;
    }
}

// Load trajectory pair from backend
function loadTrajectoryPair() {
    console.log(`LOAD_TRAJECTORY_PAIR called. IsLoading: ${isLoading}, HasError: ${hasError}, PairIndex: ${currentPairIndex}`);
    if (isLoading || hasError) {
        console.log('LoadTrajectoryPair: Skipping due to loading or error state.');
                return;
            }
            
    try {
        isLoading = true;
        updateUI();  // Update UI to reflect loading state
        const selectedDataset = currentDataset;
        
        // Hide preference summary if it exists
        const preferenceSummary = document.getElementById('preference-summary');
        if (preferenceSummary) preferenceSummary.classList.add('hidden');
        
        const url = `${BACKEND_URL}/api/get-trajectory-pair?pair_index=${currentPairIndex}&dataset=${selectedDataset}`;
        console.log('LoadTrajectoryPair: Fetching from URL:', url);
        
        fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Origin': window.location.origin,
                'Content-Type': 'application/json'
            },
            mode: 'cors',
            redirect: 'error'
        })
        .then(response => {
            if (!response.ok) {
                let errorDetail = `Server responded with ${response.status}: ${response.statusText} for ${url}`;
                return response.text().then(text => {
                    throw new Error(`${errorDetail} - Body: ${text}`);
                }).catch(() => {
                    throw new Error(errorDetail);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (!data.trajectory_a_url || !data.trajectory_b_url || !data.segments) {
                throw new Error('Server returned incomplete trajectory data');
            }

            console.log('LoadTrajectoryPair: Successfully received trajectory data');
            loadVideosForCurrentPair(data.trajectory_a_url, data.trajectory_b_url);
            
            // Update segment display
            const segmentAElement = document.getElementById('segment-a');
            const segmentBElement = document.getElementById('segment-b');
            if (segmentAElement) segmentAElement.textContent = data.segments[0];
            if (segmentBElement) segmentBElement.textContent = data.segments[1];
            
            // Update reward display
            const rewardAElement = document.getElementById('reward-a');
            const rewardBElement = document.getElementById('reward-b');
            if (rewardAElement) rewardAElement.textContent = data.reward_a.toFixed(4);
            if (rewardBElement) rewardBElement.textContent = data.reward_b.toFixed(4);
            
            isLoading = false;
            updateUI();  // Update UI after loading completes
        })
        .catch(error => {
            console.error('LoadTrajectoryPair: Error fetching or processing trajectory pair:', error.message, error.stack);
            showError(error);
            isLoading = false;
            updateUI();  // Update UI after error
        });
        
        } catch (error) {
        console.error('LoadTrajectoryPair: Synchronous error:', error.message, error.stack);
        showError(error);
        isLoading = false;
        updateUI();  // Update UI after error
    }
}

// Load videos for current pair
function loadVideosForCurrentPair(urlA, urlB) {
    console.log('LOAD_VIDEOS_FOR_CURRENT_PAIR called.');
    if (hasError) {
        console.log('LoadVideos: Aborting due to error state.');
        return;
    }

    // Disable preference buttons while loading
    if (elements.preferAButton) elements.preferAButton.disabled = true;
    if (elements.preferEqualButton) elements.preferEqualButton.disabled = true;
    if (elements.preferBButton) elements.preferBButton.disabled = true;

    Promise.all([
        loadVideo(elements.trajectoryAVideo, urlA),
        loadVideo(elements.trajectoryBVideo, urlB)
    ]).then(() => {
        if (hasError) {
            console.log('LoadVideos: Error state after videos loaded/failed. Not enabling buttons or playing.');
            return;
        }
        
        console.log('LoadVideos: Both videos loaded successfully.');
        // Enable preference buttons
        if (elements.preferAButton) elements.preferAButton.disabled = false;
        if (elements.preferEqualButton) elements.preferEqualButton.disabled = false;
        if (elements.preferBButton) elements.preferBButton.disabled = false;

        // Start playing both videos
        console.log('LoadVideos: Attempting to play videos.');
        if (elements.trajectoryAVideo) {
            const playPromiseA = elements.trajectoryAVideo.play();
            if (playPromiseA !== undefined) {
                playPromiseA.catch(error => {
                    console.error('LoadVideos: Error playing trajectory A:', error.message, error.stack);
                    showError(new Error(`Error playing video A: ${error.message}`));
                });
            }
        }
        if (elements.trajectoryBVideo) {
            const playPromiseB = elements.trajectoryBVideo.play();
            if (playPromiseB !== undefined) {
                playPromiseB.catch(error => {
                    console.error('LoadVideos: Error playing trajectory B:', error.message, error.stack);
                    showError(new Error(`Error playing video B: ${error.message}`));
                });
            }
        }
    }).catch(error => {
        console.error('LoadVideos: Error in Promise.all for loading videos:', error.message, error.stack);
        if (!hasError) {
            showError(new Error(`Failed to load one or both videos: ${error.message}`));
        }
    });
    }
    
    // Record user preference
async function recordPreference(preference) {
    if (isLoading) return;
    isLoading = true;
    updateUI(); // Update UI to show loading state

    try {
        // Save preference to the regular preferences array
        preferences.push({
            pair_index: currentPairIndex,
            preference: preference,
            timestamp: new Date().toISOString()
        });

        // Save preferences to backend
        await savePreferences();

        // Reset loading state before moving to next pair
        isLoading = false;
        
        // Load a random pair instead of just incrementing
        currentPairIndex = getRandomPairIndex();
        console.log(`Preference recorded, loading random pair: ${currentPairIndex}`);
        loadTrajectoryPair();
    } catch (error) {
        console.error('Error recording preference:', error);
        showError('Error recording preference: ' + error.message);
        isLoading = false;
        updateUI(); // Update UI after error
    }
}
    
// Save preferences to backend
async function savePreferences() {
        try {
            const submission = {
                session_id: sessionId,
            dataset: currentDataset,
                preferences: preferences,
                submitted_at: new Date().toISOString(),
                user_agent: navigator.userAgent
            };
            
        console.log('Saving preferences:', submission);
        
        const response = await safeFetch(`${BACKEND_URL}/api/save-preferences`, {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json'
                },
                body: JSON.stringify(submission)
            });
            
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }
            
        const result = await response.json();
        console.log('Preferences saved successfully:', result);
            
        // Update dataset info after saving
        await updateDatasetInfo();
        } catch (error) {
        console.error('Error saving preferences:', error);
        showError('Error saving preferences: ' + error.message);
        }
    }
    
// Navigation functions
    function showPreviousPair() {
    console.log('SHOW_PREVIOUS_PAIR called.');
    if (isLoading || hasError) {
        console.log('ShowPreviousPair: Skipping due to loading or error state.');
        return;
    }

        if (currentPairIndex > 0) {
            currentPairIndex--;
        console.log(`ShowPreviousPair: Moving to pair index ${currentPairIndex}`);
        loadTrajectoryPair();
    }
}

function showNextPair() {
    console.log('SHOW_NEXT_PAIR called.');
    if (isLoading || hasError) {
        console.log('ShowNextPair: Skipping due to loading or error state.');
        return;
    }

    currentPairIndex++;
    console.log(`ShowNextPair: Moving to pair index ${currentPairIndex}`);
    loadTrajectoryPair();
}

// Function to update UI state
function updateUI() {
    console.log('UPDATE_UI called');
    
    // Update acquisition select
    if (elements.acquisitionSelect) elements.acquisitionSelect.disabled = isLoading || hasError;
    
    // Update preference buttons (regular preference collection - should work independently)
    const buttonsDisabled = isLoading || hasError;
    if (elements.preferAButton) elements.preferAButton.disabled = buttonsDisabled;
    if (elements.preferEqualButton) elements.preferEqualButton.disabled = buttonsDisabled;
    if (elements.preferBButton) elements.preferBButton.disabled = buttonsDisabled;
    
    // Update pair index displays
    if (elements.pairIndexInput) {
        elements.pairIndexInput.value = currentPairIndex;
        elements.pairIndexInput.disabled = isLoading || hasError;
    }
    if (elements.currentPairIndexDisplay) {
        elements.currentPairIndexDisplay.textContent = currentPairIndex;
    }
    if (elements.goToPairButton) {
        elements.goToPairButton.disabled = isLoading || hasError;
    }

    // Update active learning UI
    if (elements.startActiveSessionButton) {
        elements.startActiveSessionButton.disabled = isLoading || hasError || isActiveSession;
    }
    if (elements.maxIterationsSelect) {
        elements.maxIterationsSelect.disabled = isLoading || hasError || isActiveSession;
    }
    
    // Navigation buttons are disabled during active session only if we're in the active learning tab
    const currentTab = document.querySelector('.tab-content.active');
    const isActiveTab = currentTab && currentTab.id === 'active-tab';
    
    if (elements.prevPairButton) elements.prevPairButton.disabled = (isActiveTab && isActiveSession) || currentPairIndex === 0 || isLoading || hasError;
    if (elements.nextPairButton) elements.nextPairButton.disabled = (isActiveTab && isActiveSession) || isLoading || hasError;
    if (elements.randomPairButton) elements.randomPairButton.disabled = (isActiveTab && isActiveSession) || isLoading || hasError;
}

function restartSession() {
    console.log('RESTART_SESSION called.');
    hasError = false;
    initialize();
}

// Switch between tabs
function switchTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content and activate button
    const selectedTab = document.getElementById(tabId);
    const selectedButton = document.querySelector(`[data-tab="${tabId}"]`);
    
    if (selectedTab && selectedButton) {
        selectedTab.classList.add('active');
        selectedButton.classList.add('active');
        
        // If switching to active learning tab and in active session, load the next pair
        if (tabId === 'active-tab' && isActiveSession) {
            loadNextActivePair();
        }
        
        // If switching to similar segments tab, load a random segment
        if (tabId === 'similar-tab') {
            loadRandomSimilarSegment();
        }
    } else {
        console.error(`Tab elements not found for ID: ${tabId}`);
    }
}

// Function to load a random similar segment
function loadRandomSimilarSegment() {
    // Generate a random segment index (0 to total_segments-1)
    const maxSegmentIndex = datasetInfo.total_segments > 0 ? datasetInfo.total_segments - 1 : 999;
    const randomIndex = Math.floor(Math.random() * (maxSegmentIndex + 1));
    
    console.log(`Generated random segment index: ${randomIndex} (max available: ${maxSegmentIndex})`);
    
    // Set the input value
    if (elements.segmentIndexInput) {
        elements.segmentIndexInput.value = randomIndex;
    }
    
    // Automatically trigger the search
    findSimilarSegments();
}

// Function to find similar segments
async function findSimilarSegments() {
    try {
        const segmentIndex = parseInt(elements.segmentIndexInput.value);
        const k = parseInt(elements.kValueInput.value) || 5;
        const dtwType = elements.dtwTypeSelect ? elements.dtwTypeSelect.value : 'dtw';
        
        if (isNaN(segmentIndex)) {
            showError('Please enter a valid segment index');
            return;
        }
        
        // Clear previous results
        elements.similarSegmentsContainer.innerHTML = '';
        elements.dissimilarSegmentsContainer.innerHTML = '';
        if (elements.targetSegmentVideo) {
            elements.targetSegmentVideo.src = '';
        }
        
        // Show loading state
        elements.findSimilarButton.disabled = true;
        elements.findSimilarButton.textContent = 'Loading...';
        
        const response = await safeFetch(`${BACKEND_URL}/api/get-similar-segments?dataset=${currentDataset}&segment_index=${segmentIndex}&k=${k}&dtw_type=${dtwType}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Load target segment video
        if (data.target) {
            // Configure video properties
            elements.targetSegmentVideo.loop = true;
            elements.targetSegmentVideo.muted = true;
            elements.targetSegmentVideo.playsInline = true;
            elements.targetSegmentVideo.preload = 'auto';
            
            // Set up video loading with autoplay
            elements.targetSegmentVideo.addEventListener('loadeddata', () => {
                console.log(`Target video loaded: ${data.target.video_url}`);
                elements.targetSegmentVideo.play().catch(error => {
                    console.error(`Error playing target video:`, error);
                });
            }, { once: true }); // Use once to avoid duplicate listeners
            
            elements.targetSegmentVideo.src = data.target.video_url;
            elements.targetSegmentIndex.textContent = data.target.segment_index;
            elements.targetSegmentReward.textContent = data.target.reward.toFixed(3);
        }
        
        // Create cards for similar segments
        data.similar.forEach(segment => {
            const card = createSegmentCard(segment, 'similar');
            elements.similarSegmentsContainer.appendChild(card);
        });
        
        // Create cards for dissimilar segments
        data.dissimilar.forEach(segment => {
            const card = createSegmentCard(segment, 'dissimilar');
            elements.dissimilarSegmentsContainer.appendChild(card);
        });
        
    } catch (error) {
        console.error('Error finding similar segments:', error);
        showError(`Error finding similar segments: ${error.message}`);
    } finally {
        elements.findSimilarButton.disabled = false;
        elements.findSimilarButton.textContent = 'Find Similar Segments';
    }
}

function createSegmentCard(segment, type) {
    const card = document.createElement('div');
    card.className = `similar-segment-card ${type}`;
    
    // Create video container
    const videoContainer = document.createElement('div');
    videoContainer.className = 'video-container';
    
    const video = document.createElement('video');
    video.controls = true;
    video.loop = true;
    video.muted = true;
    video.autoplay = true;
    video.playsInline = true;
    video.preload = 'auto';
    
    // Set up video loading with error handling
    video.addEventListener('loadeddata', () => {
        console.log(`Segment video loaded: ${segment.video_url}`);
        video.play().catch(error => {
            console.error(`Error playing segment video ${segment.segment_index}:`, error);
        });
    });
    
    video.addEventListener('error', (error) => {
        console.error(`Error loading segment video ${segment.segment_index}:`, error);
        // Show a placeholder or error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'video-error';
        errorDiv.textContent = 'Video not available';
        videoContainer.replaceChild(errorDiv, video);
    });
    
    video.src = segment.video_url;
    videoContainer.appendChild(video);
    
    // Create info section
    const infoSection = document.createElement('div');
    infoSection.className = 'segment-info-section';
    
    const info = document.createElement('div');
    info.className = 'segment-info';
    
    const segmentInfo = document.createElement('p');
    segmentInfo.innerHTML = `<strong>Segment:</strong> <span class="segment-number">${segment.segment_index}</span>`;
    
    const dtwType = elements.dtwTypeSelect ? elements.dtwTypeSelect.value.toUpperCase() : 'DTW';
    const distanceInfo = document.createElement('p');
    distanceInfo.innerHTML = `<strong>${dtwType}:</strong> <span class="distance-value">${segment.distance.toFixed(3)}</span>`;
    
    const rewardInfo = document.createElement('p');
    rewardInfo.innerHTML = `<strong>Reward:</strong> <span class="reward-value">${segment.reward !== undefined ? segment.reward.toFixed(3) : 'N/A'}</span>`;
    
    info.appendChild(segmentInfo);
    info.appendChild(distanceInfo);
    info.appendChild(rewardInfo);
    
    infoSection.appendChild(info);
    
    // Assemble the card (side-by-side layout)
    card.appendChild(videoContainer);
    card.appendChild(infoSection);
    
    return card;
}

// Function to go to specific pair
function goToPair() {
    const newIndex = parseInt(elements.pairIndexInput.value);
    if (isNaN(newIndex) || newIndex < 0) {
        showError(new Error('Please enter a valid pair index'));
        return;
    }
    currentPairIndex = newIndex;
    loadTrajectoryPair();
}

// Function to handle acquisition method change
function handleAcquisitionChange(event) {
    currentAcquisition = event.target.value;
    
    // Show/hide acquisition info based on method
    const acquisitionInfo = document.getElementById('acquisition-info');
    if (acquisitionInfo) {
        if (currentAcquisition === 'random') {
            acquisitionInfo.classList.add('hidden');
        } else {
            acquisitionInfo.classList.remove('hidden');
        }
    }
    
    // Load next pair with new acquisition method
    loadNextPair();
}

// Function to load the next pair based on current acquisition method
async function loadNextPair() {
    try {
        const response = await safeFetch(`${BACKEND_URL}/api/get-next-pair?dataset=${currentDataset}&acquisition=${currentAcquisition}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        // Update pair index and load the pair
        currentPairIndex = data.pair_index;
        
        // Update UI elements
        if (elements.currentPairIndexDisplay) {
            elements.currentPairIndexDisplay.textContent = currentPairIndex;
        }
        const unlabeledCountDisplay = document.getElementById('unlabeled-count');
        if (unlabeledCountDisplay) {
            unlabeledCountDisplay.textContent = data.unlabeled_pairs;
        }
        if (elements.pairIndexInput) {
            elements.pairIndexInput.value = currentPairIndex;
        }
        
        // Update acquisition score if available
        const acquisitionScore = document.getElementById('acquisition-score');
        const acquisitionInfo = document.getElementById('acquisition-info');
        if (acquisitionScore && data.acquisition_score !== null) {
            acquisitionScore.textContent = data.acquisition_score.toFixed(4);
            if (acquisitionInfo) acquisitionInfo.classList.remove('hidden');
        } else if (acquisitionInfo) {
            acquisitionInfo.classList.add('hidden');
        }
        
        // Load the trajectory pair
        loadTrajectoryPair();
        
        // Update dataset info
        document.getElementById('labeled-count').textContent = data.labeled_pairs;
        document.getElementById('total-count').textContent = data.total_pairs;
        document.getElementById('progress-percent').textContent = 
            Math.round((data.labeled_pairs / data.total_pairs) * 100);
            
    } catch (error) {
        console.error('Error loading next pair:', error);
        showError(error);
    }
}

// Function to load next active pair
async function loadNextActivePair() {
    try {
        const sessionId = activeSessionId || sessionId;
        const response = await safeFetch(`${BACKEND_URL}/api/get-next-pair?dataset=${currentDataset}&acquisition=${currentAcquisition}&session_id=${sessionId}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        // Update pair index and load the pair
        currentPairIndex = data.pair_index;
        
        // Update UI elements
        const activeCurrentPairIndexDisplay = document.getElementById('active-current-pair-index');
        if (activeCurrentPairIndexDisplay) {
            activeCurrentPairIndexDisplay.textContent = currentPairIndex;
        }
        const acquisitionScore = document.getElementById('acquisition-score');
        const acquisitionInfo = document.getElementById('acquisition-info');
        if (acquisitionScore && data.acquisition_score !== null) {
            acquisitionScore.textContent = data.acquisition_score.toFixed(4);
            if (acquisitionInfo) acquisitionInfo.classList.remove('hidden');
        }
        
        // Load the trajectory pair
        loadActiveTrajectoryPair();
        
        // Update dataset info
        document.getElementById('active-labeled-count').textContent = data.labeled_pairs;
        document.getElementById('active-total-count').textContent = data.total_pairs;
        document.getElementById('active-progress-percent').textContent = 
            Math.round((data.labeled_pairs / data.total_pairs) * 100);
        document.getElementById('active-unlabeled-count').textContent = data.unlabeled_pairs;
            
    } catch (error) {
        console.error('Error loading next active pair:', error);
        showError(error);
    }
}

// Function to load active trajectory pair
function loadActiveTrajectoryPair() {
    if (isLoading || hasError) return;

    try {
        isLoading = true;
        updateUI();

        const url = `${BACKEND_URL}/api/get-trajectory-pair?pair_index=${currentPairIndex}&dataset=${currentDataset}`;
        
        fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            mode: 'cors'
        })
        .then(response => {
            if (!response.ok) throw new Error(`Server responded with ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.error) throw new Error(data.error);
            
            if (!data.trajectory_a_url || !data.trajectory_b_url || !data.segments) {
                throw new Error('Server returned incomplete trajectory data');
            }

            loadVideosForActivePair(data.trajectory_a_url, data.trajectory_b_url);
            
            // Update segment display
            const activeSegmentAElement = document.getElementById('active-segment-a');
            const activeSegmentBElement = document.getElementById('active-segment-b');
            if (activeSegmentAElement) activeSegmentAElement.textContent = data.segments[0];
            if (activeSegmentBElement) activeSegmentBElement.textContent = data.segments[1];
            
            // Update reward display
            const activeRewardAElement = document.getElementById('active-reward-a');
            const activeRewardBElement = document.getElementById('active-reward-b');
            if (activeRewardAElement) activeRewardAElement.textContent = data.reward_a.toFixed(4);
            if (activeRewardBElement) activeRewardBElement.textContent = data.reward_b.toFixed(4);
            
            isLoading = false;
            updateUI();
        })
        .catch(error => {
            console.error('Error loading active trajectory pair:', error);
            showError(error);
            isLoading = false;
            updateUI();
        });
        
    } catch (error) {
        console.error('Error in loadActiveTrajectoryPair:', error);
        showError(error);
        isLoading = false;
        updateUI();
    }
}

// Function to load videos for active pair
function loadVideosForActivePair(urlA, urlB) {
    if (hasError) return;

    // Disable preference buttons while loading
    if (elements.activePreferAButton) elements.activePreferAButton.disabled = true;
    if (elements.activePreferEqualButton) elements.activePreferEqualButton.disabled = true;
    if (elements.activePreferBButton) elements.activePreferBButton.disabled = true;

    Promise.all([
        loadVideo(elements.activeTrajectoryAVideo, urlA),
        loadVideo(elements.activeTrajectoryBVideo, urlB)
    ]).then(() => {
        if (hasError) return;
        
        // Enable preference buttons
        if (elements.activePreferAButton) elements.activePreferAButton.disabled = false;
        if (elements.activePreferEqualButton) elements.activePreferEqualButton.disabled = false;
        if (elements.activePreferBButton) elements.activePreferBButton.disabled = false;

        // Start playing both videos
        if (elements.activeTrajectoryAVideo) {
            const playPromiseA = elements.activeTrajectoryAVideo.play();
            if (playPromiseA !== undefined) {
                playPromiseA.catch(error => {
                    console.error('Error playing active trajectory A:', error);
                    showError(new Error(`Error playing video A: ${error.message}`));
                });
            }
        }
        if (elements.activeTrajectoryBVideo) {
            const playPromiseB = elements.activeTrajectoryBVideo.play();
            if (playPromiseB !== undefined) {
                playPromiseB.catch(error => {
                    console.error('Error playing active trajectory B:', error);
                    showError(new Error(`Error playing video B: ${error.message}`));
                });
            }
        }
    }).catch(error => {
        console.error('Error loading active videos:', error);
        if (!hasError) {
            showError(new Error(`Failed to load one or both videos: ${error.message}`));
        }
    });
}

// Function to record active preference
async function recordActivePreference(preference) {
    if (isLoading) return;
    isLoading = true;

    try {
        // Show training popup and disable buttons
        showTrainingPopup('Recording preference and training reward model...');
        
        // Disable active preference buttons during training
        if (elements.activePreferAButton) elements.activePreferAButton.disabled = true;
        if (elements.activePreferEqualButton) elements.activePreferEqualButton.disabled = true;
        if (elements.activePreferBButton) elements.activePreferBButton.disabled = true;
        
        const response = await safeFetch(`${BACKEND_URL}/api/active-preferences`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                taskId: currentDataset,
                pairIndex: currentPairIndex,
                preference: preference,
                acquisitionMethod: elements.acquisitionSelect.value,
                timestamp: new Date().toISOString(),
                sessionId: activeSessionId,
                maxIterations: maxIterations,
                currentIteration: currentSession ? currentSession.currentIteration : 0
            })
        });

        if (!response.ok) {
            hideTrainingPopup();
            throw new Error('Failed to record active preference');
        }

        const data = await response.json();
        
        // Show success message if available, then hide training popup
        if (data.message) {
            showTrainingPopup(data.message, false, true); // No spinner, mark as success
            // Hide after 3 seconds to give user time to read
            setTimeout(() => {
                hideTrainingPopup();
            }, 3000);
        } else {
            hideTrainingPopup();
        }
        
        // Re-enable active preference buttons
        if (elements.activePreferAButton) elements.activePreferAButton.disabled = false;
        if (elements.activePreferEqualButton) elements.activePreferEqualButton.disabled = false;
        if (elements.activePreferBButton) elements.activePreferBButton.disabled = false;
        
        if (data.session_complete) {
            showError('Active learning session complete!');
            return;
        }

        // Increment current iteration
        if (currentSession) {
            currentSession.currentIteration++;
        }
        currentIteration++;

        // Update progress
        updateActiveProgress();
        
        // Load next pair
        await loadNextActivePair();
        
        // Update metadata after successful training
        updateActiveMetadata(data);
    } catch (error) {
        console.error('Error recording active preference:', error);
        hideTrainingPopup(); // Hide training popup on error
        
        // Re-enable active preference buttons on error
        if (elements.activePreferAButton) elements.activePreferAButton.disabled = false;
        if (elements.activePreferEqualButton) elements.activePreferEqualButton.disabled = false;
        if (elements.activePreferBButton) elements.activePreferBButton.disabled = false;
        
        showError('Error recording active preference: ' + error.message);
    } finally {
        isLoading = false;
    }
}

// Start new active learning session
async function startNewActiveSession() {
    try {
        console.log('Starting new active learning session...');
        const dataset = document.getElementById('dataset-select').value;
        const maxIterations = parseInt(document.getElementById('max-iterations').value);
        const sessionId = Date.now().toString();
        
        console.log('Session parameters:', { dataset, maxIterations, sessionId });

        const response = await safeFetch(`${BACKEND_URL}/api/start-active-session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset,
                maxIterations,
                sessionId
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Set active session state
            isActiveSession = true;
            activeSessionId = sessionId;
            currentIteration = 0; // Reset iteration counter
            
            // Reset active learning counters
            document.getElementById('active-labeled-count').textContent = '0';
            document.getElementById('active-unlabeled-count').textContent = document.getElementById('active-total-count').textContent;
            
            // Show progress bar and other UI elements
            document.getElementById('activeProgress').classList.remove('hidden');
            document.getElementById('maxIterationDisplay').textContent = maxIterations;
            document.getElementById('currentIteration').textContent = '0';
            document.getElementById('progressFill').style.width = '0%';
            
            // Show active learning UI elements
            const activePairInfo = document.getElementById('active-pair-info');
            const activeTrajectories = document.getElementById('active-trajectories');
            const activePreferenceButtons = document.getElementById('active-preference-buttons');
            
            if (activePairInfo) activePairInfo.classList.remove('hidden');
            if (activeTrajectories) activeTrajectories.classList.remove('hidden');
            if (activePreferenceButtons) activePreferenceButtons.classList.remove('hidden');

            // Store session info
            currentSession = {
                id: sessionId,
                maxIterations,
                currentIteration: 0
            };

            // Hide metadata section initially
            const metadataSection = document.getElementById('active-metadata');
            if (metadataSection) {
                metadataSection.classList.add('hidden');
            }

            // Load first pair
            await loadNextActivePair();
        } else {
            showError(data.error || 'Failed to start active learning session');
        }
    } catch (error) {
        showError('Error starting active session: ' + error.message);
    }
}

// Update active learning progress display
function updateActiveProgress() {
    const currentIterationDisplay = document.getElementById('currentIteration');
    if (currentIterationDisplay) {
        currentIterationDisplay.textContent = currentIteration;
    }
    const progressFill = document.getElementById('progressFill');
    if (progressFill) {
        const progress = maxIterations > 0 ? (currentIteration / maxIterations) * 100 : 0;
        progressFill.style.width = `${progress}%`;
    }
}

// Update active learning metadata
async function updateActiveMetadata(trainingData) {
    try {
        // Show metadata section
        const metadataSection = document.getElementById('active-metadata');
        if (metadataSection) {
            metadataSection.classList.remove('hidden');
        }
        
        // Update plots (the training metrics are now shown in the plot itself)
        await updatePlots();
        
    } catch (error) {
        console.error('Error updating metadata:', error);
    }
}

// Update matplotlib plots
async function updatePlots() {
    try {
        const sessionId = activeSessionId || sessionId;
        
        // Update acquisition score distribution plot
        try {
            const acquisitionPlotUrl = `${BACKEND_URL}/api/acquisition-plot?dataset=${currentDataset}&acquisition=${currentAcquisition}&session_id=${sessionId}&t=${Date.now()}`;
            const acquisitionImg = document.getElementById('acquisition-plot');
            if (acquisitionImg) {
                // Add error handling for image loading
                acquisitionImg.onerror = function() {
                    console.error('Failed to load acquisition plot');
                    this.style.display = 'none';
                };
                acquisitionImg.onload = function() {
                    console.log('Acquisition plot loaded successfully');
                    this.style.display = 'block';
                };
                acquisitionImg.src = acquisitionPlotUrl;
            }
            

        } catch (error) {
            console.log('No acquisition scores available yet');
        }
        
        // // Update training progress plot
        // try {
        //     const trainingPlotUrl = `${BACKEND_URL}/api/training-plot?dataset=${currentDataset}&session_id=${sessionId}&t=${Date.now()}`;
        //     const trainingImg = document.getElementById('training-plot');
        //     if (trainingImg) {
        //         // Add error handling for image loading
        //         trainingImg.onerror = function() {
        //             console.error('Failed to load training plot');
        //             this.style.display = 'none';
        //         };
        //         trainingImg.onload = function() {
        //             console.log('Training plot loaded successfully');
        //             this.style.display = 'block';
        //         };
        //         trainingImg.src = trainingPlotUrl;
        //     }
        // } catch (error) {
        //     console.log('No training data available yet');
        // }
        
    } catch (error) {
        console.error('Error updating plots:', error);
    }
}

window.showError = showError;