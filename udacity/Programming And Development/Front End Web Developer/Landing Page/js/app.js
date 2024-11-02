document.addEventListener('DOMContentLoaded', function() {
  // Get all the sections in the document
  const sections = document.querySelectorAll('section');

  // Create the navigation menu dynamically
  const navList = document.getElementById('navbar__list');
  sections.forEach(function(section) {
    const navItem = document.createElement('li');
    const navLink = document.createElement('a');
    navLink.textContent = section.getAttribute('data-nav');
    navLink.setAttribute('href', `#${section.id}`);
    navItem.appendChild(navLink);
    navList.appendChild(navItem);
  });

  // Toggle the navigation menu on small screens
  const navbarToggle = document.querySelector('.navbar__toggle');
  const navbarMenu = document.querySelector('.navbar__menu');

  navbarToggle.addEventListener('click', function() {
    navbarMenu.classList.toggle('open');
    navbarToggle.classList.toggle('open');
  });

  // Set the active section and corresponding navigation item
  function setActiveSection() {
    sections.forEach(function(section) {
      const rect = section.getBoundingClientRect();
      if (rect.top >= 0 && rect.top <= window.innerHeight * 0.5) {
        section.classList.add('your-active-class');
        const navLink = document.querySelector(`a[href="#${section.id}"]`);
        navLink.classList.add('active');
      } else {
        section.classList.remove('your-active-class');
        const navLink = document.querySelector(`a[href="#${section.id}"]`);
        navLink.classList.remove('active');
      }
    });
  }

  // Scroll to the appropriate section when a navigation item is clicked
  navList.addEventListener('click', function(event) {
    event.preventDefault();
    if (event.target.tagName === 'A') {
      const targetSection = document.querySelector(event.target.getAttribute('href'));
      targetSection.scrollIntoView({ behavior: 'smooth' });
    }
  });

  // Add event listener for scrolling and set the active section
  window.addEventListener('scroll', setActiveSection);
  
  // Hide the menu when a navigation item is clicked on smaller screens
  const menuLinks = document.querySelectorAll('.navbar__menu a');
  menuLinks.forEach(function(link) {
    link.addEventListener('click', function() {
      if (window.innerWidth <= 769) {
        navbarMenu.classList.remove('open');
      }
    });
  });
});
