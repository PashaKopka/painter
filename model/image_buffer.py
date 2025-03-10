import random

import torch


class ImageBuffer:
    def __init__(self, max_size=50, p=0.5):
        self.max_size = max_size
        self.buffer = []
        self.probability = p

    def get_images(self, generated_images: torch.Tensor):
        return_images = []
        for img in generated_images:
            img = torch.unsqueeze(img.data, 0)

            # if buffer is not full, add images to it
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                return_images.append(img)
            else:
                if random.uniform(0, 1) > self.probability:
                    idx = random.randint(0, self.max_size - 1)
                    output_image = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    return_images.append(output_image)
                else:
                    return_images.append(img)

        return_images = torch.cat(return_images, dim=0)
        return return_images
