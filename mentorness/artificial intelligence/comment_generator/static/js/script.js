// Initialize the canvas and context
const canvas = document.getElementById('particleCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Particle properties
const particles = [];
const particleCount = 100;
const particleSize = 5;

// Particle class
class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * particleSize + 1;
        this.speedX = Math.random() * 3 - 1.5;
        this.speedY = Math.random() * 3 - 1.5;
        this.alpha = 0;
    }
    draw() {
        ctx.fillStyle = `rgba(255, 255, 255, ${this.alpha})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
    }
    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.alpha < 1) this.alpha += 0.01;
        if (this.alpha >= 1) this.alpha -= 0.01;
        if (this.size > 0.2) this.size -= 0.05;
        if (this.size <= 0.2) this.size = Math.random() * particleSize + 1;
        this.draw();
    }
}

// Create particles
function initParticles() {
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
}

// Animate particles
function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(particle => {
        particle.update();
    });
    requestAnimationFrame(animateParticles);
}

// Adjust canvas size on window resize
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    particles.length = 0;
    initParticles();
});

// Initialize and animate particles
initParticles();
animateParticles();
