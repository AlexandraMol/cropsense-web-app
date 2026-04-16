function toggleInput() {
    const source = document.getElementById("data_source").value;
    const upload = document.getElementById("upload_section");
    const mongo = document.getElementById("mongodb_section");

    if (source === "upload") {
        upload.style.display = "block";
        mongo.style.display = "none";
    } else {
        upload.style.display = "none";
        mongo.style.display = "block";
    }
}