document.addEventListener('DOMContentLoaded', () => {
    // تابع قبلی برای ترسیم نقاط
    document.querySelectorAll('.image-container').forEach(container => {
        const img = container.querySelector('img');
        const points = container.querySelectorAll('.point');
        
        img.onload = () => {
            const imgWidth = img.naturalWidth;
            const imgHeight = img.naturalHeight;
            const displayWidth = img.offsetWidth;
            const displayHeight = img.offsetHeight;

            points.forEach(point => {
                const originalX = parseFloat(point.dataset.x);
                const originalY = parseFloat(point.dataset.y);

                const displayX = (originalX / imgWidth) * displayWidth;
                const displayY = (originalY / imgHeight) * displayHeight;
                
                point.style.left = `${displayX}px`;
                point.style.top = `${displayY}px`;
            });
        };
    });

    // تابع جدید برای توصیف تصاویر
    window.describeImages = function(button) {
        const row = button.closest('.row');
        const image1Path = button.dataset.image1;
        const image2Path = button.dataset.image2;
        const descriptionArea = row.querySelector('.description-area');

        descriptionArea.textContent = 'در حال تحلیل تصاویر...';

        fetch('/describe_images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image1_path: image1Path,
                image2_path: image2Path
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                descriptionArea.textContent = data.combined_description;
            } else {
                descriptionArea.textContent = 'خطا در تحلیل تصاویر: ' + data.message;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            descriptionArea.textContent = 'خطا در ارتباط با سرور.';
        });
    };

    // تابع به‌روزرسانی کپشن (بدون تغییر)
    window.updateCaption = function(button) {
        const row = button.closest('.row');
        const jsonPath = row.dataset.jsonPath;
        const input = row.querySelector('input');
        const caption = input.value;
    
        fetch('/update_caption', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                json_file_path: jsonPath,
                caption: caption
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('کپشن با موفقیت ثبت شد!');
                // window.location.href = '/';
            } else {
                alert('خطا در ثبت کپشن: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('خطا در ارتباط با سرور.');
        });
    };
});