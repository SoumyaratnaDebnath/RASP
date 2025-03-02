function updateImagesPanda(value) {
    document.getElementById('imageSliderValuePanda').textContent = value*50;
    document.getElementById('1image2').src = `GIF_Database/Panda/GA_images/img (${value}).png`;
    document.getElementById('1image3').src = `GIF_Database/Panda/DE_images/img (${value}).png`;
    document.getElementById('1image4').src = `GIF_Database/Panda/GSA_images/img (${value}).png`;
    document.getElementById('1image5').src = `GIF_Database/Panda/PSO_images/img (${value}).png`;
    document.getElementById('1image6').src = `GIF_Database/Panda/HHO_images/img (${value}).png`;
}
function updateImagesBird(value) {
    document.getElementById('imageSliderValueBird').textContent = value*50;
    document.getElementById('2image2').src = `GIF_Database/Bird/GA_images/img (${value}).png`;
    document.getElementById('2image3').src = `GIF_Database/Bird/DE_images/img (${value}).png`;
    document.getElementById('2image4').src = `GIF_Database/Bird/GSA_images/img (${value}).png`;
    document.getElementById('2image5').src = `GIF_Database/Bird/PSO_images/img (${value}).png`;
    document.getElementById('2image6').src = `GIF_Database/Bird/HHO_images/img (${value}).png`;
}
function updateImagesHand(value) {
    document.getElementById('imageSliderValueHand').textContent = value*50;
    document.getElementById('3image2').src = `GIF_Database/hand/GA_images/img (${value}).png`;
    document.getElementById('3image3').src = `GIF_Database/hand/DE_images/img (${value}).png`;
    document.getElementById('3image4').src = `GIF_Database/hand/GSA_images/img (${value}).png`;
    document.getElementById('3image5').src = `GIF_Database/hand/PSO_images/img (${value}).png`;
    document.getElementById('3image6').src = `GIF_Database/hand/HHO_images/img (${value}).png`;
}
function updateImagesHut(value) {
    document.getElementById('imageSliderValueHut').textContent = value*50;
    document.getElementById('5image2').src = `GIF_Database/hut/GA_images/img (${value}).png`;
    document.getElementById('5image3').src = `GIF_Database/hut/DE_images/img (${value}).png`;
    document.getElementById('5image4').src = `GIF_Database/hut/GSA_images/img (${value}).png`;
    document.getElementById('5image5').src = `GIF_Database/hut/PSO_images/img (${value}).png`;
    document.getElementById('5image6').src = `GIF_Database/hut/HHO_images/img (${value}).png`;
}
function updateImagesRose(value) {
    document.getElementById('imageSliderValueRose').textContent = value*50;
    document.getElementById('6image2').src = `GIF_Database/rose/GA_images/img (${value}).png`;
    document.getElementById('6image3').src = `GIF_Database/rose/DE_images/img (${value}).png`;
    document.getElementById('6image4').src = `GIF_Database/rose/GSA_images/img (${value}).png`;
    document.getElementById('6image5').src = `GIF_Database/rose/PSO_images/img (${value}).png`;
    document.getElementById('6image6').src = `GIF_Database/rose/HHO_images/img (${value}).png`;
}

// function updateImages(value) {
//     document.getElementById('imageSliderValue').textContent = value*50;
//     document.getElementById('1image2').src = `GIF_Database/Panda/GA_images/img (${value}).png`;
//     document.getElementById('1image3').src = `GIF_Database/Panda/DE_images/img (${value}).png`;
//     document.getElementById('1image4').src = `GIF_Database/Panda/GSA_images/img (${value}).png`;
//     document.getElementById('1image5').src = `GIF_Database/Panda/PSO_images/img (${value}).png`;
//     document.getElementById('1image6').src = `GIF_Database/Panda/HHO_images/img (${value}).png`;

//     document.getElementById('2image2').src = `GIF_Database/Bird/GA_images/img (${value}).png`;
//     document.getElementById('2image3').src = `GIF_Database/Bird/DE_images/img (${value}).png`;
//     document.getElementById('2image4').src = `GIF_Database/Bird/GSA_images/img (${value}).png`;
//     document.getElementById('2image5').src = `GIF_Database/Bird/PSO_images/img (${value}).png`;
//     document.getElementById('2image6').src = `GIF_Database/Bird/HHO_images/img (${value}).png`;

//     document.getElementById('3image2').src = `GIF_Database/hand/GA_images/img (${value}).png`;
//     document.getElementById('3image3').src = `GIF_Database/hand/DE_images/img (${value}).png`;
//     document.getElementById('3image4').src = `GIF_Database/hand/GSA_images/img (${value}).png`;
//     document.getElementById('3image5').src = `GIF_Database/hand/PSO_images/img (${value}).png`;
//     document.getElementById('3image6').src = `GIF_Database/hand/HHO_images/img (${value}).png`;

//     document.getElementById('4image2').src = `GIF_Database/horse/GA_images/img (${value}).png`;
//     document.getElementById('4image3').src = `GIF_Database/horse/DE_images/img (${value}).png`;
//     document.getElementById('4image4').src = `GIF_Database/horse/GSA_images/img (${value}).png`;
//     document.getElementById('4image5').src = `GIF_Database/horse/PSO_images/img (${value}).png`;
//     document.getElementById('4image6').src = `GIF_Database/horse/HHO_images/img (${value}).png`;

//     document.getElementById('5image2').src = `GIF_Database/hut/GA_images/img (${value}).png`;
//     document.getElementById('5image3').src = `GIF_Database/hut/DE_images/img (${value}).png`;
//     document.getElementById('5image4').src = `GIF_Database/hut/GSA_images/img (${value}).png`;
//     document.getElementById('5image5').src = `GIF_Database/hut/PSO_images/img (${value}).png`;
//     document.getElementById('5image6').src = `GIF_Database/hut/HHO_images/img (${value}).png`;

//     document.getElementById('6image2').src = `GIF_Database/rose/GA_images/img (${value}).png`;
//     document.getElementById('6image3').src = `GIF_Database/rose/DE_images/img (${value}).png`;
//     document.getElementById('6image4').src = `GIF_Database/rose/GSA_images/img (${value}).png`;
//     document.getElementById('6image5').src = `GIF_Database/rose/PSO_images/img (${value}).png`;
//     document.getElementById('6image6').src = `GIF_Database/rose/HHO_images/img (${value}).png`;

//     document.getElementById('7image2').src = `GIF_Database/Landscape/GA_images/img (${value}).png`;
//     document.getElementById('7image3').src = `GIF_Database/Landscape/DE_images/img (${value}).png`;
//     document.getElementById('7image4').src = `GIF_Database/Landscape/GSA_images/img (${value}).png`;
//     document.getElementById('7image5').src = `GIF_Database/Landscape/PSO_images/img (${value}).png`;
//     document.getElementById('7image6').src = `GIF_Database/Landscape/HHO_images/img (${value}).png`;

//     document.getElementById('8image2').src = `GIF_Database/statue/GA_images/img (${value}).png`;
//     document.getElementById('8image3').src = `GIF_Database/statue/DE_images/img (${value}).png`;
//     document.getElementById('8image4').src = `GIF_Database/statue/GSA_images/img (${value}).png`;
//     document.getElementById('8image5').src = `GIF_Database/statue/PSO_images/img (${value}).png`;
//     document.getElementById('8image6').src = `GIF_Database/statue/HHO_images/img (${value}).png`;
// }

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('placeHolder').style.height = document.getElementById('teaserImage').clientHeight + 'px';

    window.addEventListener('scroll', function() {
        var scrollPosition = window.scrollY;
        var tiltX = (scrollPosition / 100) % 20; // Horizontal tilt
        var tiltY = (scrollPosition / 100) % 15; // Vertical tilt
        var translateY = scrollPosition / -2; // Vertical parallax effect
        var scale = 1 + scrollPosition / 2000; // Slightly zoom in

        document.getElementById('teaserImage').style.transform = `translateY(${translateY}px) rotateX(${tiltY}deg) rotateY(${tiltX}deg) scale(${scale})`;
        document.getElementById('teaserImage').style.opacity = 1 - scrollPosition / 1000; // Keep some fade for smooth transition
        document.getElementById('teaserImage').style.transition = 'transform 0.1s ease-out'; // Smooth animation
    });

});