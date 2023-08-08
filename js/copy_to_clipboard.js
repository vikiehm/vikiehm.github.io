function formateCitation(citation) {
  const reg = /,([^=]+=)/g
  var new_format = citation.replace(reg, ",\n  $1")

  new_format = new_format.replace(/,( *)}/g, ",$1\n}")
  return new_format
}

function formateCitationHTML(citation) {
  const reg = /,([^=]+=)/g
  var new_format = citation.replace(reg, ",<br>  $1")

  new_format = new_format.replace(/,( *)}/g, ",$1<br>}")
  return new_format
}

function tempAlert(msg, duration) {
  // make more beautiful
  var el = document.createElement("div");
  el.setAttribute("style", "position:fixed;top:20px;left:20%;right:20%;background-color:white;border-radius:10px;border-style:solid;border-width:3px;border-color:black;padding:3px;");
  el.innerHTML = msg;
  // el.innerHTML = msg;
  setTimeout(function () {
    el.parentNode.removeChild(el);
  }, duration);
  document.body.appendChild(el);
}

function CopyToClipboard(copyText) {
  var citation = formateCitation(copyText)
  // Copy the text inside the text field
  navigator.clipboard.writeText(citation);

  // Alert the copied text
  tempAlert('Copied Citation:<br>' + formateCitationHTML(copyText), 2000);
}