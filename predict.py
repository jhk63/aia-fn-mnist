
import cv2
import torch
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
# from PIL import ImageOps

from model import Net


plt.switch_backend('Agg')

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image

def extract_digits(contours, image):
    digits = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

    for box in bounding_boxes:
        x, y, w, h = box
        digit = image[y:y+h, x:x+w]
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit = cv2.bitwise_not(digit)  # 이미지 색상 반전(흰 -> 검)
        digits.append(digit)

    print(len(digits))
    return digits

def predict_digits(digits, net):
    predictions = []

    for digit in digits:
        tensor = to_tensor(digit).unsqueeze(0)
        output = net(tensor)
        _, predicted = torch.max(output.data, 1)
        predictions.append(predicted.item())

    return predictions

def visualize_contours(image, contours):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.show()

def visualize_sequence(image, contours, save_path='uploads/output_contours.png'):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def visualize_digits(digits):
    num_digits = len(digits)
    plt.figure(figsize=(10, 2))

    for i, digit in enumerate(digits):
        plt.subplot(1, num_digits, i + 1)
        plt.imshow(digit, cmap='gray')
        plt.axis('off')

    plt.show()

def main(image_path):
    contours, image = preprocess_image(image_path)
    visualize_contours(image, contours)

    digits = extract_digits(contours, image)
    print(f"Number of digits detected: {len(digits)}")
    # visualize_digits(digits)

    net = Net()
    net.load_state_dict(torch.load('mnist_cnn3.pth'))
    net.eval()

    predictions = predict_digits(digits, net)
    print("Predicted digits:", predictions)

if __name__ == '__main__':
    image_path = 'Test_Image/68.png'
    # image_path = 'Test_Image/786.png'
    # image_path = 'Test_Image/48.png'
    # image_path = 'Test_Image/34.png'
    main(image_path)
