// static/js/app.js - Book Recommendation System

document.addEventListener("DOMContentLoaded", () => {
    initializeApp();
});

/**
 * Main application initialization
 */
function initializeApp() {
    initializeTheme();
    initializeMenu();
    initializeFormEnhancements();
}

/**
 * Theme management with system preference detection
 */
function initializeTheme() {
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;
    
    // Get saved theme or detect system preference
    const savedTheme = localStorage.getItem("theme");
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Set initial theme
    if (savedTheme === "dark-mode" || (!savedTheme && systemPrefersDark)) {
        body.classList.add("dark-mode");
        themeToggle.textContent = "🌙";
    } else {
        themeToggle.textContent = "☀️";
    }
    
    // Theme toggle handler
    themeToggle.addEventListener("click", () => {
        body.classList.toggle("dark-mode");
        const isDark = body.classList.contains("dark-mode");
        localStorage.setItem("theme", isDark ? "dark-mode" : "light-mode");
        themeToggle.textContent = isDark ? "🌙" : "☀️";
    });
}

/**
 * Menu management
 */
function initializeMenu() {
    const menuToggle = document.getElementById("menu-toggle");
    const menuPanel = document.getElementById("menu-panel");
    
    if (!menuToggle || !menuPanel) return;
    
    // Click handler for menu toggle
    menuToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        menuPanel.classList.toggle("open");
    });
    
    // Close menu when clicking outside
    document.addEventListener("click", (e) => {
        if (menuPanel.classList.contains("open") && 
            !menuPanel.contains(e.target) && 
            e.target !== menuToggle) {
            menuPanel.classList.remove("open");
        }
    });
    
    // Close menu with Escape key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && menuPanel.classList.contains("open")) {
            menuPanel.classList.remove("open");
        }
    });
}

/**
 * Form enhancements for better user experience
 */
function initializeFormEnhancements() {
    // Enhanced range inputs
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(range => {
        range.addEventListener('input', updateRangeValue);
        // Initialize values
        updateRangeValue.call(range);
    });
    
    // Auto-submit forms when certain inputs change
    const autoSubmitInputs = document.querySelectorAll('select[name="order_by"], select[name="page_size"]');
    autoSubmitInputs.forEach(input => {
        input.addEventListener('change', () => {
            if (input.closest('form')) {
                // Reset to page 1 when changing sort or page size
                const pageInput = input.closest('form').querySelector('input[name="page"]');
                if (pageInput) {
                    pageInput.value = 1;
                }
                input.closest('form').submit();
            }
        });
    });
    
    // Auto-submit when book is selected in Smart Match
    const bookSelect = document.getElementById("selected_book");
    if (bookSelect) {
        bookSelect.addEventListener('change', function() {
            if (this.value) {
                this.closest('form').submit();
            }
        });
    }
}

/**
 * Update range input value display
 */
function updateRangeValue() {
    const range = this;
    const valueElement = document.getElementById(`${range.id}_value`) || 
                        range.parentNode.querySelector('span') ||
                        document.querySelector(`span[id*="${range.id}"]`);
    
    if (valueElement) {
        valueElement.textContent = range.value;
    }
}

/**
 * Utility function for pagination navigation
 */
function goToPage(newPage) {
    const form = document.getElementById('quick-pick-form');
    if (form) {
        const pageInput = document.createElement('input');
        pageInput.type = 'hidden';
        pageInput.name = 'page';
        pageInput.value = newPage;
        form.appendChild(pageInput);
        form.submit();
    }
}