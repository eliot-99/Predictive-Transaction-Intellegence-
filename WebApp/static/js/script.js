/**
 * FraudGuard BFSI - Main JavaScript
 * Handles client-side interactivity and utilities
 */

// =====================================================
// DOM Ready
// =====================================================

document.addEventListener('DOMContentLoaded', function() {
    initializePopovers();
    initializeTooltips();
    setupEventListeners();
    checkAPIHealth();
});

// =====================================================
// Bootstrap Popovers & Tooltips
// =====================================================

function initializePopovers() {
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// =====================================================
// Event Listeners
// =====================================================

function setupEventListeners() {
    // Auto-close alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Smooth scroll for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href !== '#' && document.querySelector(href)) {
                e.preventDefault();
                document.querySelector(href).scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

// =====================================================
// API Health Check
// =====================================================

async function checkAPIHealth() {
    try {
        // Check if we're on a page that needs API connectivity
        if (!document.querySelector('[data-requires-api]')) {
            return;
        }

        const response = await fetch('/api/health', {
            method: 'GET',
            timeout: 5000
        });

        if (!response.ok) {
            showAPIWarning('Fraud detection API is temporarily unavailable');
        }
    } catch (error) {
        console.warn('API health check failed:', error);
        // Don't show warning for connectivity issues on non-essential pages
    }
}

function showAPIWarning(message) {
    const alertHtml = `
        <div class="alert alert-warning alert-dismissible fade show" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>API Warning:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    const container = document.querySelector('main');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alertHtml);
    }
}

// =====================================================
// Number Formatting
// =====================================================

function formatCurrency(value, currency = 'UZS') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency === 'USD' ? 'USD' : 'EUR'
    }).format(value);
}

function formatNumber(value, decimals = 0) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatPercentage(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

// =====================================================
// Date & Time Utilities
// =====================================================

function formatDateTime(date) {
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    return new Date(date).toLocaleDateString('en-US', options);
}

function formatTimeAgo(date) {
    const seconds = Math.floor((new Date() - new Date(date)) / 1000);
    
    const intervals = {
        year: 31536000,
        month: 2592000,
        week: 604800,
        day: 86400,
        hour: 3600,
        minute: 60
    };

    for (const [name, secondsInInterval] of Object.entries(intervals)) {
        const interval = Math.floor(seconds / secondsInInterval);
        if (interval >= 1) {
            return interval === 1 ? `1 ${name} ago` : `${interval} ${name}s ago`;
        }
    }

    return 'just now';
}

// =====================================================
// Local Storage Utilities
// =====================================================

const Storage = {
    set: function(key, value) {
        try {
            localStorage.setItem(`fg_${key}`, JSON.stringify(value));
        } catch (e) {
            console.warn('localStorage unavailable:', e);
        }
    },

    get: function(key) {
        try {
            const item = localStorage.getItem(`fg_${key}`);
            return item ? JSON.parse(item) : null;
        } catch (e) {
            console.warn('localStorage unavailable:', e);
            return null;
        }
    },

    remove: function(key) {
        try {
            localStorage.removeItem(`fg_${key}`);
        } catch (e) {
            console.warn('localStorage unavailable:', e);
        }
    },

    clear: function() {
        try {
            const keys = Object.keys(localStorage);
            keys.forEach(key => {
                if (key.startsWith('fg_')) {
                    localStorage.removeItem(key);
                }
            });
        } catch (e) {
            console.warn('localStorage unavailable:', e);
        }
    }
};

// =====================================================
// Notification System
// =====================================================

const Notification = {
    show: function(message, type = 'info', duration = 3000) {
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';

        const alertHtml = `
            <div class="alert ${alertClass} alert-dismissible fade show animate__animated animate__fadeIn" role="alert">
                <i class="fas fa-${type === 'error' ? 'exclamation-circle' : type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        const container = document.querySelector('main') || document.body;
        const alertElement = document.createElement('div');
        alertElement.innerHTML = alertHtml;
        container.insertBefore(alertElement.firstElementChild, container.firstChild);

        if (duration > 0) {
            setTimeout(() => {
                const alert = container.querySelector('.alert');
                if (alert) {
                    const bsAlert = new bootstrap.Alert(alert);
                    bsAlert.close();
                }
            }, duration);
        }
    },

    success: function(message, duration = 3000) {
        this.show(message, 'success', duration);
    },

    error: function(message, duration = 5000) {
        this.show(message, 'error', duration);
    },

    warning: function(message, duration = 4000) {
        this.show(message, 'warning', duration);
    },

    info: function(message, duration = 3000) {
        this.show(message, 'info', duration);
    }
};

// =====================================================
// Loading Spinner
// =====================================================

const LoadingSpinner = {
    show: function(message = 'Loading...') {
        const spinnerHtml = `
            <div id="loadingSpinner" class="position-fixed top-50 start-50 translate-middle z-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-center mt-3 text-muted">${message}</p>
            </div>
            <div id="spinnerBackdrop" class="position-fixed top-0 start-0 w-100 h-100 bg-dark z-4" style="opacity: 0.3;"></div>
        `;
        
        const container = document.createElement('div');
        container.innerHTML = spinnerHtml;
        document.body.appendChild(container.firstElementChild);
        document.body.appendChild(container.firstElementChild);
    },

    hide: function() {
        const spinner = document.getElementById('loadingSpinner');
        const backdrop = document.getElementById('spinnerBackdrop');
        
        if (spinner) spinner.remove();
        if (backdrop) backdrop.remove();
    }
};

// =====================================================
// Form Utilities
// =====================================================

function resetFormFields(formId) {
    const form = document.getElementById(formId);
    if (form) {
        form.reset();
    }
}

function disableFormSubmit(formId) {
    const form = document.getElementById(formId);
    if (form) {
        const buttons = form.querySelectorAll('button[type="submit"]');
        buttons.forEach(btn => {
            btn.disabled = true;
            btn.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>Processing...`;
        });
    }
}

function enableFormSubmit(formId) {
    const form = document.getElementById(formId);
    if (form) {
        const buttons = form.querySelectorAll('button[type="submit"]');
        buttons.forEach(btn => {
            btn.disabled = false;
            btn.textContent = btn.dataset.originalText || 'Submit';
        });
    }
}

// =====================================================
// Export to CSV
// =====================================================

function exportToCSV(filename, data) {
    let csv = '\ufeff'; // BOM for Excel

    // Add headers
    if (data.length > 0) {
        const headers = Object.keys(data[0]);
        csv += headers.map(h => `"${h}"`).join(',') + '\n';

        // Add data rows
        data.forEach(row => {
            const values = headers.map(h => {
                const value = row[h];
                // Escape quotes and handle numbers
                return `"${String(value).replace(/"/g, '""')}"`;
            });
            csv += values.join(',') + '\n';
        });
    }

    // Create download link
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// =====================================================
// Chart Utilities
// =====================================================

function getRiskScoreColor(score) {
    if (score >= 0.7) return '#dc3545'; // Red - High Risk
    if (score >= 0.4) return '#ffc107'; // Yellow - Medium Risk
    return '#28a745'; // Green - Low Risk
}

function getRiskScoreLabel(score) {
    if (score >= 0.7) return 'HIGH RISK';
    if (score >= 0.4) return 'MEDIUM RISK';
    return 'SAFE';
}

// =====================================================
// API Helpers
// =====================================================

async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API error:', error);
        Notification.error('Failed to fetch data. Please try again.');
        throw error;
    }
}

// =====================================================
// Validation Utilities
// =====================================================

const Validation = {
    email: function(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },

    phone: function(phone) {
        const re = /^[0-9]{10,}$/;
        return re.test(phone.replace(/\D/g, ''));
    },

    creditCard: function(cc) {
        return /^[0-9]{13,19}$/.test(cc.replace(/\s/g, ''));
    },

    password: function(password) {
        return password.length >= 6;
    },

    required: function(value) {
        return value && value.trim().length > 0;
    }
};

// =====================================================
// Keyboard Shortcuts
// =====================================================

document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + K for quick search
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        const searchInput = document.querySelector('[data-quick-search]');
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Escape key to close modals
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        });
    }
});

// =====================================================
// Performance Monitoring
// =====================================================

const Performance = {
    startTime: null,
    marks: {},

    start: function(label = 'default') {
        this.marks[label] = performance.now();
    },

    end: function(label = 'default') {
        if (!this.marks[label]) return null;
        
        const duration = performance.now() - this.marks[label];
        console.log(`[${label}] completed in ${duration.toFixed(2)}ms`);
        
        delete this.marks[label];
        return duration;
    }
};

// =====================================================
// Console Helpers (Development)
// =====================================================

if (typeof process !== 'undefined' && process.env.NODE_ENV !== 'production') {
    window.fgDebug = {
        storage: Storage,
        notification: Notification,
        validation: Validation,
        performance: Performance
    };
}

// =====================================================
// Global Error Handler
// =====================================================

window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    // Could send to error tracking service like Sentry
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
});

console.log('FraudGuard BFSI - JavaScript loaded successfully');