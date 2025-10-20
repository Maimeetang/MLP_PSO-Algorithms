def transpose(lst):
    return list(map(list, zip(*lst)))

class normalizer:
    
    def normalize_sample(self, sample_list):
        """
        normalize function
        """
        
        sample_list = transpose(sample_list)
        input_list, output_list = sample_list
        
        input_list_t = transpose(input_list)
        output_list_t = transpose(output_list)

        self.input_mean_list = []
        self.input_std_list = []
        input_norm_list_t = []

        for i in range(len(input_list_t)):
            mean = sum(input_list_t[i]) / len(input_list_t[i])
            variance = sum((x - mean) ** 2 for x in input_list_t[i]) / len(input_list_t[i])
            std = variance ** 0.5

            self.input_mean_list.append(mean)
            self.input_std_list.append(std)
            input_norm_list_t.append([(x - mean) / std for x in input_list_t[i]])

        self.output_mean_list = []
        self.output_std_list = []
        output_norm_list_t = []

        for i in range(len(output_list_t)):
            mean = sum(output_list_t[i]) / len(output_list_t[i])
            variance = sum((x - mean) ** 2 for x in output_list_t[i]) / len(output_list_t[i])
            std = variance ** 0.5

            self.output_mean_list.append(mean)
            self.output_std_list.append(std)
            output_norm_list_t.append([(x - mean) / std for x in output_list_t[i]])

        input_norm_list = transpose(input_norm_list_t)
        output_norm_list = transpose(output_norm_list_t)
        sample_norm_list = [input_norm_list,output_norm_list]

        return transpose(sample_norm_list)

    def normalize_validation_input(self, val_input):
        """
        ใช้ค่า mean และ std จาก training (normalize_sample) 
        เพื่อ normalize validation input
        """
        if not hasattr(self, "input_mean_list") or not hasattr(self, "input_std_list"):
            raise ValueError("ต้องเรียก normalize_sample ก่อน เพื่อคำนวณ mean/std")

        if len(val_input) != len(self.input_mean_list):
            raise ValueError(f"validation input ต้องมี {len(self.input_mean_list)} ค่า")

        norm_input = []
        for x, mean, std in zip(val_input, self.input_mean_list, self.input_std_list):
            z = (x - mean) / std if std != 0 else 0
            norm_input.append(z)
        return norm_input
    
    def denormalize_output(self, norm_output):
        """แปลงค่าผลลัพธ์ normalized กลับเป็นค่าจริง"""
        if not hasattr(self, "output_mean_list") or not hasattr(self, "output_std_list"):
            raise ValueError("ต้องเรียก normalize_sample ก่อน เพื่อคำนวณ mean/std")

        if len(norm_output) != len(self.output_mean_list):
            raise ValueError(f"output ต้องมี {len(self.output_mean_list)} ค่า")

        return [
            z * std + mean
            for z, mean, std in zip(norm_output, self.output_mean_list, self.output_std_list)
        ]