window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    svg: {
        fontCache: 'global'
    },
    chtml: {
        scale: 1,                      // global scaling factor for all expressions
        minScale: .5,                  // smallest scaling factor to use
        matchFontHeight: true,         // true to match ex-height of surrounding font
        mtextInheritFont: true,        // true to make mtext elements use surrounding font
        merrorInheritFont: false,      // true to make merror text use surrounding font
        merrorFont: 'serif',           // font to use for merror, if not inheriting (empty means use MathJax fonts)
        mathmlSpacing: false,          // true for MathML spacing rules, false for TeX rules
        skipAttributes: {},            // RFDa and other attributes NOT to copy to the output
        exFactor: .5,                  // default size of ex in em units
        displayAlign: 'center',        // default for indentalign when set to 'auto'
        displayIndent: '0'             // default for indentshift when set to 'auto'
    }
};
(function () {
    var script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-svg.js";
    script.async = true;
    document.head.appendChild(script);
})();