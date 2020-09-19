window.MathJax = {
    TeX: {
        equationNumbers: {
            autoNumber: "AMS",
        },
    }
};

// This is for hiding unnecessary elements when displayed in the FRDS app
if (navigator.userAgent.indexOf("FRDS") !== -1) {
    // Hide all
    var classList = document.getElementsByTagName("*");
    classList.forEach(element => {
        element.style.display = "none";
    });
    // Display only the main article
    var article = document.getElementsByTagName("article")[0];
    article.style.display = "block";
}