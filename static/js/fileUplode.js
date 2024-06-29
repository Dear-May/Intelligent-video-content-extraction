$(document).ready(function () {
    function handleFileUpload(fileInput, progressId, container) {
        var file = fileInput.files[0];
        if (file) {
            // 显示进度条并重置其状态
            $(progressId).removeClass('d-none');
            $(`${progressId} .progress-bar`).css('width', '0%').attr('aria-valuenow', 0);

            // 创建 FormData 对象
            var formData = new FormData();
            formData.append('video', file);

            // AJAX 文件上传
            $.ajax({
                url: '/upload-video',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                xhr: function () {
                    var xhr = new XMLHttpRequest();
                    // 处理进度事件
                    xhr.upload.addEventListener('progress', function (e) {
                        if (e.lengthComputable) {
                            var percentComplete = (e.loaded / e.total) * 100;
                            $(`${progressId} .progress-bar`).css('width', percentComplete + '%').attr('aria-valuenow', percentComplete);
                        }
                    }, false);
                    return xhr;
                },
                success: function (response) {
                    // 上传成功，替换页面内容为视频播放器
                    var videoUrl = response.fileUrl;  // 假设服务器返回视频的URL
                    var videoPlayerHtml = `
                        <div class="bg-dark p-4 rounded-3 d-flex flex-column align-items-center justify-content-center w-100 h-100" style="border-style: dashed;">
                            <video controls class="w-100" style="height: 100%;">
                                <source src="${videoUrl}" type="video/mp4" id="original-video">
                                您的浏览器不支持 HTML5 视频播放。
                            </video>
                        </div>
                    `;
                    $(container).html(videoPlayerHtml);
                    if (typeof successCallback === 'function') {
                        successCallback();
                    }
                },
                error: function () {
                    alert('上传失败，请重试。');
                }
            });
        }
    }

    $('#upload-button').on('click', function () {
        $('#video-upload').click();
    });

    $('#video-upload').on('change', function (event) {
        handleFileUpload(this, '#upload-progress', '.video-container');
    });

    $('#upload-button1').on('click', function () {
        $('#video-upload1').click();
    });

    $('#video-upload1').on('change', function (event) {
        handleFileUpload(this, '#upload-progress', '.video-container');
    });
});