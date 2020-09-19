window.MathJax = {
    TeX: {
        equationNumbers: {
            autoNumber: "AMS",
        },
    }
};

// This is for hiding unnecessary elements when displayed in the FRDS app
if (navigator.userAgent.indexOf("FRDS") !== -1) {
    var elems = document.getElementsByTagName("*");
    var article = document.getElementsByTagName("article")[0];
    for (var i = 0; i < elems.length; i++) {
        var elem = elems[i];
        if (!article.contains(elem) && !elem.contains(article)) {
            elem.style.display = "none"
        }
    }
    var h1 = document.getElementsByTagName("h1")[0];
    h1.style.display = "none";
    var icons = document.getElementsByClassName("md-icon");
    for (var i = 0; i < icons.length; i++) {
        icons[i].style.display = "none";
    }
}