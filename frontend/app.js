/**
 * Tesla Stock Forecasting Frontend Application
 * Handles user interactions, API calls, and data visualization
 */

class TeslaForecastApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000';
        this.chart = null;
        this.currentPrediction = null;
        this.isServerReady = false;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateTimeHorizon();
        this.checkServerStatus();
        this.loadHistory();
        
        // Check server status every 30 seconds
        setInterval(() => this.checkServerStatus(), 30000);
    }

    bindEvents() {
        // Slider and input synchronization
        const slider = document.getElementById('intervalSlider');
        const input = document.getElementById('intervalCount');
        
        slider.addEventListener('input', (e) => {
            input.value = e.target.value;
            this.updateTimeHorizon();
        });
        
        input.addEventListener('input', (e) => {
            const value = Math.min(Math.max(e.target.value, 1), 24);
            e.target.value = value;
            slider.value = value;
            this.updateTimeHorizon();
        });

        // Predict button
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.generatePrediction();
        });

        // Refresh buttons
        document.getElementById('refreshBtn').addEventListener('click', () => {
            if (this.currentPrediction) {
                this.generatePrediction();
            }
        });

        document.getElementById('historyRefreshBtn').addEventListener('click', () => {
            this.loadHistory();
        });

        // Export button
        document.getElementById('exportBtn').addEventListener('click', () => {
            this.exportData();
        });

        // Chart type change
        document.getElementById('chartType').addEventListener('change', (e) => {
            if (this.currentPrediction) {
                this.renderChart(this.currentPrediction, e.target.value);
            }
        });

        // Modal close events
        document.getElementById('modalClose').addEventListener('click', () => {
            this.hideModal();
        });

        document.getElementById('errorOkBtn').addEventListener('click', () => {
            this.hideModal();
        });

        // Close modal on background click
        document.getElementById('errorModal').addEventListener('click', (e) => {
            if (e.target.id === 'errorModal') {
                this.hideModal();
            }
        });
    }

    updateTimeHorizon() {
        const intervals = parseInt(document.getElementById('intervalCount').value);
        const minutes = intervals * 15;
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = minutes % 60;
        
        let timeText;
        if (hours > 0) {
            timeText = remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours} hour${hours > 1 ? 's' : ''}`;
        } else {
            timeText = `${minutes} minutes`;
        }
        
        document.getElementById('timeHorizon').textContent = timeText;
        document.getElementById('timeHorizonDetails').textContent = `${intervals} intervals × 15 minutes`;
    }

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (response.ok && data.predictor_ready) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Connected & Ready';
                this.isServerReady = true;
                document.getElementById('predictBtn').disabled = false;
            } else {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Server Initializing...';
                this.isServerReady = false;
                document.getElementById('predictBtn').disabled = true;
            }
        } catch (error) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Connection Error';
            this.isServerReady = false;
            document.getElementById('predictBtn').disabled = true;
        }
    }

    async generatePrediction() {
        if (!this.isServerReady) {
            this.showError('Server is not ready. Please wait for initialization to complete.');
            return;
        }

        const intervals = parseInt(document.getElementById('intervalCount').value);
        
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiBaseUrl}/predict?intervals=${intervals}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate prediction');
            }
            
            this.currentPrediction = data;
            this.renderPredictionResults(data);
            this.renderChart(data);
            this.loadMetrics();
            this.loadHistory(); // Refresh history after new prediction
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError(`Failed to generate prediction: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    renderPredictionResults(data) {
        const resultsContent = document.getElementById('resultsContent');
        
        // Calculate summary statistics
        const arimaPredictions = data.predictions.map(p => p.arima_prediction);
        const sarimaPredictions = data.predictions.map(p => p.sarima_prediction);
        
        const arimaAvg = arimaPredictions.reduce((a, b) => a + b) / arimaPredictions.length;
        const sarimaAvg = sarimaPredictions.reduce((a, b) => a + b) / sarimaPredictions.length;
        
        const arimaChange = ((arimaPredictions[arimaPredictions.length - 1] - data.current_price) / data.current_price * 100);
        const sarimaChange = ((sarimaPredictions[sarimaPredictions.length - 1] - data.current_price) / data.current_price * 100);
        
        resultsContent.innerHTML = `
            <div class="prediction-summary fade-in">
                <div class="summary-card">
                    <div class="value">$${data.current_price.toFixed(2)}</div>
                    <div class="label">Current Price</div>
                </div>
                <div class="summary-card">
                    <div class="value">$${arimaAvg.toFixed(2)}</div>
                    <div class="label">ARIMA Avg</div>
                </div>
                <div class="summary-card">
                    <div class="value">$${sarimaAvg.toFixed(2)}</div>
                    <div class="label">SARIMA Avg</div>
                </div>
                <div class="summary-card">
                    <div class="value ${arimaChange >= 0 ? 'text-success' : 'text-error'}">${arimaChange >= 0 ? '+' : ''}${arimaChange.toFixed(2)}%</div>
                    <div class="label">ARIMA Change</div>
                </div>
            </div>
            
            <div class="prediction-table slide-up">
                <table>
                    <thead>
                        <tr>
                            <th>Time Ahead</th>
                            <th>ARIMA Prediction</th>
                            <th>SARIMA Prediction</th>
                            <th>Confidence Range</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.predictions.map(pred => {
                            const confidenceWidth = pred.arima_confidence_interval.upper - pred.arima_confidence_interval.lower;
                            return `
                                <tr>
                                    <td><strong>${pred.minutes_ahead} min</strong><br>
                                        <small>${new Date(pred.timestamp).toLocaleTimeString()}</small>
                                    </td>
                                    <td class="price">$${pred.arima_prediction.toFixed(2)}</td>
                                    <td class="price">$${pred.sarima_prediction.toFixed(2)}</td>
                                    <td class="confidence">±$${confidenceWidth.toFixed(2)}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        // Show charts and metrics panels
        document.getElementById('chartsPanel').style.display = 'block';
        document.getElementById('metricsPanel').style.display = 'block';
    }

    renderChart(data, chartType = 'line') {
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            // Fallback: Display a simple text-based chart
            const chartContainer = document.getElementById('forecastChart').parentElement;
            chartContainer.innerHTML = `
                <div class="chart-fallback">
                    <h3>Prediction Visualization</h3>
                    <p class="chart-note">Chart.js library not available. Showing data summary:</p>
                    <div class="chart-data">
                        ${data.predictions.map((pred, i) => {
                            const arimaChange = ((pred.arima_prediction - data.current_price) / data.current_price * 100);
                            const sarimaChange = ((pred.sarima_prediction - data.current_price) / data.current_price * 100);
                            return `
                                <div class="chart-item">
                                    <div class="chart-time">+${pred.minutes_ahead}min</div>
                                    <div class="chart-prices">
                                        <div class="chart-price arima">
                                            ARIMA: $${pred.arima_prediction.toFixed(2)}
                                            <span class="change ${arimaChange >= 0 ? 'positive' : 'negative'}">
                                                (${arimaChange >= 0 ? '+' : ''}${arimaChange.toFixed(2)}%)
                                            </span>
                                        </div>
                                        <div class="chart-price sarima">
                                            SARIMA: $${pred.sarima_prediction.toFixed(2)}
                                            <span class="change ${sarimaChange >= 0 ? 'positive' : 'negative'}">
                                                (${sarimaChange >= 0 ? '+' : ''}${sarimaChange.toFixed(2)}%)
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
            return;
        }

        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (this.chart) {
            this.chart.destroy();
        }
        
        const labels = data.predictions.map(p => {
            const date = new Date(p.timestamp);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        });
        
        const arimaData = data.predictions.map(p => p.arima_prediction);
        const sarimaData = data.predictions.map(p => p.sarima_prediction);
        const arimaUpper = data.predictions.map(p => p.arima_confidence_interval.upper);
        const arimaLower = data.predictions.map(p => p.arima_confidence_interval.lower);
        
        const datasets = [];
        
        if (chartType === 'comparison') {
            datasets.push(
                {
                    label: 'ARIMA Prediction',
                    data: arimaData,
                    borderColor: '#e31837',
                    backgroundColor: 'rgba(227, 24, 55, 0.1)',
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'SARIMA Prediction',
                    data: sarimaData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: false,
                    tension: 0.4
                }
            );
        } else {
            // Default to SARIMA for single model view
            datasets.push({
                label: 'SARIMA Prediction',
                data: sarimaData,
                borderColor: '#e31837',
                backgroundColor: chartType === 'area' ? 'rgba(227, 24, 55, 0.2)' : 'rgba(227, 24, 55, 0.1)',
                fill: chartType === 'area',
                tension: 0.4
            });
            
            // Add confidence interval
            datasets.push({
                label: 'Confidence Interval',
                data: arimaUpper,
                borderColor: 'rgba(227, 24, 55, 0.3)',
                backgroundColor: 'transparent',
                fill: '+1',
                pointRadius: 0,
                borderWidth: 1
            });
            
            datasets.push({
                label: '',
                data: arimaLower,
                borderColor: 'rgba(227, 24, 55, 0.3)',
                backgroundColor: 'rgba(227, 24, 55, 0.1)',
                fill: false,
                pointRadius: 0,
                borderWidth: 1
            });
        }
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Tesla Stock Price Forecast',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            filter: function(legendItem) {
                                return legendItem.text !== '';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Price ($)'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    async loadMetrics() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/metrics`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to load metrics');
            }
            
            this.renderMetrics(data);
            
        } catch (error) {
            console.error('Metrics error:', error);
        }
    }

    renderMetrics(data) {
        const metricsGrid = document.getElementById('metricsGrid');
        
        const metrics = [
            {
                name: 'ARIMA RMSE',
                value: `$${data.model_performance.arima.rmse.toFixed(2)}`,
                description: 'Root Mean Square Error'
            },
            {
                name: 'ARIMA MAE',
                value: `$${data.model_performance.arima.mae.toFixed(2)}`,
                description: 'Mean Absolute Error'
            },
            {
                name: 'ARIMA AIC',
                value: data.model_performance.arima.aic.toFixed(2),
                description: 'Akaike Information Criterion'
            },
            {
                name: 'SARIMA RMSE',
                value: `$${data.model_performance.sarima.rmse.toFixed(2)}`,
                description: 'Root Mean Square Error'
            },
            {
                name: 'SARIMA MAE',
                value: `$${data.model_performance.sarima.mae.toFixed(2)}`,
                description: 'Mean Absolute Error'
            },
            {
                name: 'SARIMA AIC',
                value: data.model_performance.sarima.aic.toFixed(2),
                description: 'Akaike Information Criterion'
            },
            {
                name: 'Best Model',
                value: data.best_model.toUpperCase(),
                description: 'Based on AIC comparison'
            },
            {
                name: 'Data Points',
                value: data.data_summary.total_data_points.toLocaleString(),
                description: 'Training data size'
            }
        ];
        
        metricsGrid.innerHTML = metrics.map(metric => `
            <div class="metric-card fade-in">
                <div class="metric-name">${metric.name}</div>
                <div class="metric-value">${metric.value}</div>
                <div class="metric-description">${metric.description}</div>
            </div>
        `).join('');
    }

    async loadHistory() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/history?limit=10`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to load history');
            }
            
            this.renderHistory(data.requests);
            
        } catch (error) {
            console.error('History error:', error);
            document.getElementById('historyContent').innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Failed to load history</span>
                </div>
            `;
        }
    }

    renderHistory(requests) {
        const historyContent = document.getElementById('historyContent');
        
        if (!requests || requests.length === 0) {
            historyContent.innerHTML = `
                <div class="placeholder-content">
                    <i class="fas fa-clock"></i>
                    <h3>No Predictions Yet</h3>
                    <p>Your prediction history will appear here once you generate some forecasts.</p>
                </div>
            `;
            return;
        }
        
        const historyHtml = requests.map(request => {
            const date = new Date(request.timestamp);
            const timeAgo = this.getTimeAgo(date);
            
            return `
                <div class="history-item fade-in" onclick="app.viewHistoryDetails(${request.id})">
                    <div class="history-item-info">
                        <div class="history-item-time">${date.toLocaleString()}</div>
                        <div class="history-item-details">
                            ${request.intervals_requested} intervals • ${timeAgo}
                        </div>
                    </div>
                    <div class="history-item-summary">
                        ${request.summary ? `
                            <div class="history-item-prediction">
                                ARIMA: $${request.summary.arima_avg_prediction} | 
                                SARIMA: $${request.summary.sarima_avg_prediction}
                            </div>
                            <div class="history-item-confidence">${request.prediction_count} predictions</div>
                        ` : `
                            <div class="history-item-confidence">${request.prediction_count} predictions</div>
                        `}
                    </div>
                </div>
            `;
        }).join('');
        
        historyContent.innerHTML = `<div class="history-list">${historyHtml}</div>`;
    }

    async viewHistoryDetails(requestId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/history/${requestId}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to load prediction details');
            }
            
            // Convert the historical data to the same format as current predictions
            const formattedData = {
                current_price: data.predictions[0] ? data.predictions[0].arima.prediction : 0,
                predictions: data.predictions.map(pred => ({
                    timestamp: pred.prediction_timestamp,
                    minutes_ahead: pred.minutes_ahead,
                    arima_prediction: pred.arima.prediction,
                    arima_confidence_interval: {
                        lower: pred.arima.lower_bound,
                        upper: pred.arima.upper_bound
                    },
                    sarima_prediction: pred.sarima.prediction,
                    sarima_confidence_interval: {
                        lower: pred.sarima.lower_bound,
                        upper: pred.sarima.upper_bound
                    }
                }))
            };
            
            this.currentPrediction = formattedData;
            this.renderPredictionResults(formattedData);
            this.renderChart(formattedData);
            
            if (data.metrics) {
                const metricsData = {
                    model_performance: data.metrics,
                    best_model: data.metrics.best_model,
                    data_summary: {
                        total_data_points: data.summary.total_predictions
                    }
                };
                this.renderMetrics(metricsData);
            }
            
            // Show panels
            document.getElementById('chartsPanel').style.display = 'block';
            document.getElementById('metricsPanel').style.display = 'block';
            
        } catch (error) {
            console.error('History details error:', error);
            this.showError(`Failed to load prediction details: ${error.message}`);
        }
    }

    exportData() {
        if (!this.currentPrediction) {
            this.showError('No prediction data to export');
            return;
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            current_price: this.currentPrediction.current_price,
            predictions: this.currentPrediction.predictions
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `tesla_prediction_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / (1000 * 60));
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
        if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        return 'Just now';
    }

    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        const progressFill = document.getElementById('progressFill');
        
        overlay.classList.add('show');
        
        // Animate progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 20;
            if (progress > 90) progress = 90;
            progressFill.style.width = `${progress}%`;
        }, 200);
        
        // Store interval for cleanup
        overlay.progressInterval = interval;
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        const progressFill = document.getElementById('progressFill');
        
        // Complete progress bar
        progressFill.style.width = '100%';
        
        setTimeout(() => {
            overlay.classList.remove('show');
            progressFill.style.width = '0%';
            
            // Clear interval
            if (overlay.progressInterval) {
                clearInterval(overlay.progressInterval);
                delete overlay.progressInterval;
            }
        }, 500);
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorModal').classList.add('show');
    }

    hideModal() {
        document.getElementById('errorModal').classList.remove('show');
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new TeslaForecastApp();
});

// Export for global access
window.app = app;