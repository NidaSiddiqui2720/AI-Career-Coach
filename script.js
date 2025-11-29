// Mobile nav
const menuToggle = document.getElementById('menu-toggle');
const navLinks = document.getElementById('nav-links');
if (menuToggle && navLinks) {
  menuToggle.addEventListener('click', () => navLinks.classList.toggle('active'));
}

// Scroll animations (same as before)
const fadeElements = document.querySelectorAll('.fade-in, .feature-card');
function handleScrollAnimation() {
  fadeElements.forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight - 100) el.classList.add('visible');
  });
}
window.addEventListener('scroll', handleScrollAnimation);
handleScrollAnimation();

// ---- Contact Form handling (will post to Google Apps Script)
const contactForm = document.getElementById('contact-form');
const contactResult = document.getElementById('contact-result');

// Replace this URL after you deploy the Apps Script web app (instructions below)
const GOOGLE_SHEETS_WEBAPP_URL = "REPLACE_WITH_YOUR_APPS_SCRIPT_URL";

if (contactForm) {
  contactForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const subject = document.getElementById('subject').value.trim();
    const message = document.getElementById('message').value.trim();

    if (!name || !email || !message) {
      contactResult.textContent = "Please fill required fields.";
      return;
    }

    // If Google Apps Script endpoint is set -> POST form
    if (GOOGLE_SHEETS_WEBAPP_URL && !GOOGLE_SHEETS_WEBAPP_URL.includes("REPLACE_WITH")) {
      try {
        const res = await fetch(GOOGLE_SHEETS_WEBAPP_URL, {
          method: 'POST',
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, subject, message, timestamp: new Date().toISOString() })
        });
        const data = await res.json();
        if (data.result === 'success') {
          contactResult.textContent = "Message sent. Thank you!";
          contactForm.reset();
        } else {
          contactResult.textContent = "Server error — try again later.";
        }
      } catch (err) {
        contactResult.textContent = "Network error — check your Apps Script URL.";
      }
    } else {
      // Local demo behavior (no backend)
      contactResult.textContent = "Demo: message prepared (no backend URL set).";
      console.log({ name, email, subject, message });
    }
  });
}
