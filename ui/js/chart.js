/**
 * Smart Safety Inspector — Alert Chart (Chart.js)
 */

let alertChart = null;
let alertHistory = []; // [{timestamp, count}]

function initChart() {
    const ctx = document.getElementById('alertChart').getContext('2d');

    // Default empty data for 24 hours (hourly buckets)
    const labels = [];
    const data = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
        const h = new Date(now - i * 3600 * 1000);
        labels.push(h.getHours().toString().padStart(2, '0') + ':00');
        data.push(0);
    }

    alertChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Alerts',
                data,
                backgroundColor: 'rgba(0, 212, 255, 0.4)',
                borderColor: 'rgba(0, 212, 255, 0.8)',
                borderWidth: 1,
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#21262d',
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    borderColor: '#30363d',
                    borderWidth: 1,
                }
            },
            scales: {
                x: {
                    grid: { color: '#21262d' },
                    ticks: {
                        color: '#6e7681',
                        maxTicksLimit: 8,
                        font: { family: "'JetBrains Mono'", size: 9 }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#21262d' },
                    ticks: {
                        color: '#6e7681',
                        stepSize: 1,
                        font: { family: "'JetBrains Mono'", size: 9 }
                    }
                }
            }
        }
    });
}

function pushAlertToChart(priority) {
    if (!alertChart) return;

    const now = new Date();
    const hourLabel = now.getHours().toString().padStart(2, '0') + ':00';
    const labels = alertChart.data.labels;
    const data = alertChart.data.datasets[0].data;

    // Find current hour bucket
    const lastLabel = labels[labels.length - 1];
    if (lastLabel === hourLabel) {
        data[data.length - 1]++;
    } else {
        // Shift buckets
        labels.shift();
        labels.push(hourLabel);
        data.shift();
        data.push(1);
    }

    alertChart.update('none');
}

// Update chart from API stats
async function refreshChartFromAPI() {
    try {
        const res = await fetch('/api/alerts/stats');
        // Chart shows total count per hour — just keep local tracking
    } catch (e) {
        // Ignore
    }
}

document.addEventListener('DOMContentLoaded', initChart);
