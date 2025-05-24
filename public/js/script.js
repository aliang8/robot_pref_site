// Set this to your backend's public URL
const BACKEND_URL = 'https://snoopy1.usc.edu:8443'; // Backend server running on snoopy1 with HTTPS
console.log('SCRIPT EXECUTION STARTED. Backend URL:', BACKEND_URL); // Top-level log

// State variables
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
    unlabeled_pairs: 0
};

// DOM Element references
let trajectoryAVideo;
let trajectoryBVideo;
let preferAButton;
let preferEqualButton;
let preferBButton;
let prevPairButton;
let nextPairButton;
let thanksMessage;
let restartButton;
let datasetSelect;
let segmentAElement;
let segmentBElement;
let userPreferenceElement;
let rewardAElement;
let rewardBElement;
let preferenceSummary;
let preferredTrajectoryElement;
let agreementStatusElement;

// Additional DOM elements for similar segments tab
let tabButtons;
let tabContents;
let segmentIndexInput;
let kValueInput;
let findSimilarButton;
let targetSegmentVideo;
let similarSegmentsContainer;
let dissimilarSegmentsContainer;
let targetSegmentIndex;
let targetSegmentReward;

// Additional DOM elements
let pairIndexInput;
let goToPairButton;

// Current pair rewards
let currentRewardA = 0;
let currentRewardB = 0;

// Function to initialize DOM elements
function initializeDOMElements() {
    trajectoryAVideo = document.getElementById('trajectory-A');
    trajectoryBVideo = document.getElementById('trajectory-B');
    preferAButton = document.getElementById('prefer-A');
    preferEqualButton = document.getElementById('prefer-equal');
    preferBButton = document.getElementById('prefer-B');
    prevPairButton = document.getElementById('prev-pair');
    nextPairButton = document.getElementById('next-pair');
    thanksMessage = document.getElementById('thanks-message');
    restartButton = document.getElementById('restart-button');
    datasetSelect = document.getElementById('dataset-select');
    segmentAElement = document.getElementById('segment-a');
    segmentBElement = document.getElementById('segment-b');
    userPreferenceElement = document.getElementById('user-preference');
    rewardAElement = document.getElementById('reward-a');
    rewardBElement = document.getElementById('reward-b');
    preferenceSummary = document.getElementById('preference-summary');
    preferredTrajectoryElement = document.getElementById('preferred-trajectory');
    agreementStatusElement = document.getElementById('agreement-status');

    // New elements for pair index control
    pairIndexInput = document.getElementById('pair-index-input');
    goToPairButton = document.getElementById('go-to-pair');

    // New elements for similar segments tab
    tabButtons = document.querySelectorAll('.tab-button');
    tabContents = document.querySelectorAll('.tab-content');
    segmentIndexInput = document.getElementById('segment-index-input');
    kValueInput = document.getElementById('k-value');
    findSimilarButton = document.getElementById('find-similar');
    targetSegmentVideo = document.getElementById('target-segment-video');
    similarSegmentsContainer = document.getElementById('similar-segments-container');
    dissimilarSegmentsContainer = document.getElementById('dissimilar-segments-container');
    targetSegmentIndex = document.getElementById('target-segment-index');
    targetSegmentReward = document.getElementById('target-segment-reward');
}

// Event handler functions
let handlePreferA;
let handlePreferEqual;
let handlePreferB;
let handleVideoError;
let handleDatasetChange;

// Error handling function
function showError(error) {
    console.error('SHOW_ERROR triggered:', error.message, error.stack);
    hasError = true;
    isLoading = false;
    
    const errorPopup = document.getElementById('error-popup');
    const errorMessageElement = document.getElementById('error-message');
    const errorReload = document.getElementById('error-reload');
    const errorClose = document.getElementById('error-close');
    
    errorMessageElement.textContent = error.message || 'An unexpected error occurred.';
    errorPopup.classList.remove('hidden');
    
    errorReload.onclick = () => {
        console.log('Error popup: Reload button clicked.');
        window.location.reload();
    };
    
    errorClose.onclick = () => {
        console.log('Error popup: Close button clicked.');
        errorPopup.classList.add('hidden');
    };
}

// Function to load available datasets
async function loadAvailableDatasets() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/get-available-datasets`);
        if (!response.ok) throw new Error('Failed to fetch datasets');
        const data = await response.json();
        
        // Update dataset select
        const select = document.getElementById('dataset-select');
        select.innerHTML = data.datasets.map(dataset => 
            `<option value="${dataset}">${dataset}</option>`
        ).join('');
        
        // Set current dataset
        currentDataset = data.datasets[0];
        if (select) select.value = currentDataset;
        
        // Load initial dataset info
        await updateDatasetInfo();
    } catch (error) {
        console.error('Error loading datasets:', error);
        showError(error);
    }
}

// Function to update dataset info
async function updateDatasetInfo() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/get-dataset-info?dataset=${currentDataset}`);
        if (!response.ok) throw new Error('Failed to fetch dataset info');
        datasetInfo = await response.json();
        
        // Update UI
        document.getElementById('labeled-count').textContent = datasetInfo.labeled_pairs;
        document.getElementById('total-count').textContent = datasetInfo.total_pairs;
        document.getElementById('progress-percent').textContent = 
            Math.round((datasetInfo.labeled_pairs / datasetInfo.total_pairs) * 100);
    } catch (error) {
        console.error('Error updating dataset info:', error);
        showError(error);
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    console.log('DOMCONTENTLOADED event fired. Initializing application...');

    // Initialize DOM elements
    initializeDOMElements();
    
    // Load available datasets
    await loadAvailableDatasets();

    // Add tab switching event listeners
    tabButtons.forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });

    // Add similar segments search event listener
    if (findSimilarButton) {
        findSimilarButton.addEventListener('click', findSimilarSegments);
    }

    // Add pair index control event listeners
    if (goToPairButton) {
        goToPairButton.addEventListener('click', goToPair);
    }
    if (pairIndexInput) {
        pairIndexInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                goToPair();
            }
        });
    }

    // Prevent page refresh on form submission
    document.addEventListener('submit', function(e) {
        e.preventDefault();
    });

    // Only show unload warning if there are unsaved preferences
    window.addEventListener('beforeunload', function(e) {
        if (preferences.length > 0) {
            e.preventDefault();
            e.returnValue = 'You have unsaved preferences. Are you sure you want to leave?';
        }
    });

    // Define event handlers
    handlePreferA = () => recordPreference('A');
    handlePreferEqual = () => recordPreference('equal');
    handlePreferB = () => recordPreference('B');
    handleVideoError = (event) => {
        const videoElement = event.target;
        const error = videoElement.error;
        const errorMessage = `Video loading error on ${videoElement.id}: Code ${error.code}, Message: ${error.message}. Current src: ${videoElement.currentSrc}`;
        console.error('HANDLE_VIDEO_ERROR:', errorMessage);
        showError(new Error(errorMessage));
    };
    handleDatasetChange = async () => {
        console.log('Dataset changed to:', datasetSelect?.value);
        currentDataset = datasetSelect.value;
        await updateDatasetInfo();
        restartSession();
    };

    // Function to reset state
    function resetState() {
        console.log('RESET_STATE called.');
        preferences = [];
        isLoading = false;
        sessionId = new Date().getTime().toString();
        currentPairIndex = 0;
        hasError = false;
        currentRewardA = 0;
        currentRewardB = 0;

        // Clean up videos
        if (trajectoryAVideo) trajectoryAVideo.pause();
        if (trajectoryBVideo) trajectoryBVideo.pause();
        if (trajectoryAVideo) trajectoryAVideo.removeAttribute('src');
        if (trajectoryBVideo) trajectoryBVideo.removeAttribute('src');
        if (trajectoryAVideo) trajectoryAVideo.load();
        if (trajectoryBVideo) trajectoryBVideo.load();

        // Reset UI state
        if (thanksMessage) thanksMessage.classList.add('hidden');
        if (preferenceSummary) preferenceSummary.classList.add('hidden');
        const container = document.querySelector('.container');
        if (container) container.classList.remove('hidden');
        
        // Reset preference display
        if (userPreferenceElement) userPreferenceElement.textContent = '-';
        if (rewardAElement) rewardAElement.textContent = '-';
        if (rewardBElement) rewardBElement.textContent = '-';
        
        updateUI();
    }

    // Initialize
    function initialize() {
        console.log('INITIALIZE function started.');
        try {
            if (hasError) {
                console.log('Initialize: Aborting due to error state.');
                return;
            }
            resetState();
            loadTrajectoryPair();
        } catch (error) {
            console.error('Error in initialize function:', error.message, error.stack);
            showError(error);
        }
    }

    // Add event listeners
    if (preferAButton) preferAButton.addEventListener('click', handlePreferA);
    if (preferEqualButton) preferEqualButton.addEventListener('click', handlePreferEqual);
    if (preferBButton) preferBButton.addEventListener('click', handlePreferB);
    if (prevPairButton) prevPairButton.addEventListener('click', showPreviousPair);
    if (nextPairButton) nextPairButton.addEventListener('click', showNextPair);
    if (restartButton) restartButton.addEventListener('click', restartSession);
    if (trajectoryAVideo) trajectoryAVideo.addEventListener('error', handleVideoError);
    if (trajectoryBVideo) trajectoryBVideo.addEventListener('error', handleVideoError);
    if (datasetSelect) datasetSelect.addEventListener('change', handleDatasetChange);

    // Start the app
    initialize();
});

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
        
        // Hide preference summary
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
            if (segmentAElement) segmentAElement.textContent = data.segments[0];
            if (segmentBElement) segmentBElement.textContent = data.segments[1];
            
            // Store current rewards
            currentRewardA = data.reward_a;
            currentRewardB = data.reward_b;
            
            // Update reward display
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
    if (preferAButton) preferAButton.disabled = true;
    if (preferEqualButton) preferEqualButton.disabled = true;
    if (preferBButton) preferBButton.disabled = true;

    Promise.all([
        loadVideo(trajectoryAVideo, urlA),
        loadVideo(trajectoryBVideo, urlB)
    ]).then(() => {
        if (hasError) {
            console.log('LoadVideos: Error state after videos loaded/failed. Not enabling buttons or playing.');
            return;
        }
        
        console.log('LoadVideos: Both videos loaded successfully.');
        // Enable preference buttons
        if (preferAButton) preferAButton.disabled = false;
        if (preferEqualButton) preferEqualButton.disabled = false;
        if (preferBButton) preferBButton.disabled = false;

        // Start playing both videos
        console.log('LoadVideos: Attempting to play videos.');
        if (trajectoryAVideo) {
            const playPromiseA = trajectoryAVideo.play();
            if (playPromiseA !== undefined) {
                playPromiseA.catch(error => {
                    console.error('LoadVideos: Error playing trajectory A:', error.message, error.stack);
                    showError(new Error(`Error playing video A: ${error.message}`));
                });
            }
        }
        if (trajectoryBVideo) {
            const playPromiseB = trajectoryBVideo.play();
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
function recordPreference(preference) {
    console.log(`RECORD_PREFERENCE: Preference selected: ${preference}, IsLoading: ${isLoading}, HasError: ${hasError}`);
    if (isLoading || hasError) {
        console.log('RecordPreference: Skipping due to loading or error state.');
        return;
    }

    // Pause current videos
    if (trajectoryAVideo) trajectoryAVideo.pause();
    if (trajectoryBVideo) trajectoryBVideo.pause();

    // Update preference display
    if (userPreferenceElement) {
        userPreferenceElement.textContent = preference === 'A' ? 'Trajectory A' : 
                                          preference === 'B' ? 'Trajectory B' : 'Equal';
    }

    // Show the ground truth preference
    const groundTruthPreference = currentRewardA > currentRewardB ? 'Trajectory A' :
                                 currentRewardB > currentRewardA ? 'Trajectory B' : 'Equal';
    
    if (preferredTrajectoryElement) {
        preferredTrajectoryElement.textContent = groundTruthPreference;
    }

    // Show agreement status
    if (agreementStatusElement) {
        const userChoice = preference === 'A' ? 'Trajectory A' :
                          preference === 'B' ? 'Trajectory B' : 'Equal';
        const agrees = userChoice === groundTruthPreference;
        
        agreementStatusElement.textContent = agrees ? 
            'Your choice agrees with the ground truth!' :
            'Your choice differs from the ground truth';
        agreementStatusElement.className = 'agreement-status ' + (agrees ? 'agree' : 'disagree');
    }

    // Show the summary
    if (preferenceSummary) {
        preferenceSummary.classList.remove('hidden');
    }

    preferences.push({
        pair_index: currentPairIndex,
        preference: preference,
        timestamp: new Date().toISOString()
    });

    // Save preferences
    savePreferences();
    
    // Move to next pair after a delay
    setTimeout(() => {
        if (preferenceSummary) preferenceSummary.classList.add('hidden');
        showNextPair();
    }, 3000);
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
        
        const response = await fetch(`${BACKEND_URL}/api/save-preferences`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Origin': window.location.origin
            },
            body: JSON.stringify(submission)
        });
        
        if (!response.ok) throw new Error('Failed to save preferences');
        
        // Update dataset info after saving
        await updateDatasetInfo();
    } catch (error) {
        console.error('Error saving preferences:', error);
        showError(error);
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
    
    // Update navigation buttons
    if (prevPairButton) prevPairButton.disabled = currentPairIndex === 0 || isLoading || hasError;
    if (nextPairButton) nextPairButton.disabled = isLoading || hasError;  // Remove any limit on next button
    
    // Update preference buttons
    const buttonsDisabled = isLoading || hasError;
    if (preferAButton) preferAButton.disabled = buttonsDisabled;
    if (preferEqualButton) preferEqualButton.disabled = buttonsDisabled;
    if (preferBButton) preferBButton.disabled = buttonsDisabled;
    
    // Update pair index input
    if (pairIndexInput) {
        pairIndexInput.value = currentPairIndex;
        pairIndexInput.disabled = isLoading || hasError;
    }
    if (goToPairButton) {
        goToPairButton.disabled = isLoading || hasError;
    }
}

function restartSession() {
    console.log('RESTART_SESSION called.');
    hasError = false;
    initialize();
}

// Function to switch tabs
function switchTab(tabId) {
    tabButtons.forEach(button => {
        button.classList.remove('active');
        if (button.dataset.tab === tabId) {
            button.classList.add('active');
        }
    });

    tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${tabId}-tab`) {
            content.classList.add('active');
        }
    });
}

// Function to create a segment card
function createSegmentCard(segment, type) {
    const card = document.createElement('div');
    card.className = 'similar-segment-card';
    card.innerHTML = `
        <div class="video-container">
            <video controls loop>
                <source src="${segment.video_url}" type="video/mp4">
                Your browser does not support video playback.
            </video>
        </div>
        <div class="segment-info">
            <p>Segment Index: ${segment.segment_index}</p>
            <p>DTW Distance: <span class="dtw-distance">${segment.dtw_distance.toFixed(4)}</span></p>
            <p>Similarity Score: <span class="similarity-score">${segment.similarity.toFixed(4)}</span></p>
            <p>Reward: ${segment.reward.toFixed(4)}</p>
            <p class="segment-type ${type}">${type === 'similar' ? 'Similar' : 'Dissimilar'}</p>
        </div>
    `;
    return card;
}

// Function to find similar segments
async function findSimilarSegments() {
    const segmentIndex = segmentIndexInput.value;
    const k = kValueInput.value;
    
    if (!segmentIndex) {
        showError(new Error('Please enter a segment index'));
        return;
    }

    try {
        const response = await fetch(`${BACKEND_URL}/api/get-similar-segments?segment_index=${segmentIndex}&k=${k}&dataset=${currentDataset}`);
        if (!response.ok) throw new Error('Failed to fetch similar segments');
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        // Display target segment
        const targetSegment = data.target;
        targetSegmentVideo.src = targetSegment.video_url;
        targetSegmentVideo.load();
        targetSegmentVideo.play();
        
        targetSegmentIndex.textContent = targetSegment.segment_index;
        targetSegmentReward.textContent = targetSegment.reward.toFixed(4);
        
        // Display similar segments
        similarSegmentsContainer.innerHTML = '';
        data.similar.forEach(segment => {
            const card = createSegmentCard(segment, 'similar');
            similarSegmentsContainer.appendChild(card);
            
            // Start playing the video
            const video = card.querySelector('video');
            video.load();
            video.play();
        });

        // Display dissimilar segments
        dissimilarSegmentsContainer.innerHTML = '';
        data.dissimilar.forEach(segment => {
            const card = createSegmentCard(segment, 'dissimilar');
            dissimilarSegmentsContainer.appendChild(card);
            
            // Start playing the video
            const video = card.querySelector('video');
            video.load();
            video.play();
        });
    } catch (error) {
        console.error('Error finding similar segments:', error);
        showError(error);
    }
}

// Function to go to specific pair
function goToPair() {
    const newIndex = parseInt(pairIndexInput.value);
    if (isNaN(newIndex) || newIndex < 0) {
        showError(new Error('Please enter a valid pair index'));
        return;
    }
    currentPairIndex = newIndex;
    loadTrajectoryPair();
}

window.showError = showError;