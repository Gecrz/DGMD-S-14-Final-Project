# get all the 400x png files into a folder

# Define the tar file
tar_file_path = 'BreaKHis_v1.tar.gz'

# Define the directory to store the extracted files
output_dir = 'extracted_images_400x'
os.makedirs(output_dir, exist_ok=True)

# Open the tar file
tar = tarfile.open(tar_file_path)

# Iterate over each member in the tar file
for member in tar.getmembers():
    if '400X' in member.name and member.name.endswith('.png'):
        # Use extractfile() to get a file-like object for the file data
        file_data = tar.extractfile(member)

        if file_data is not None:
            # Get the base filename (without directories)
            base_filename = os.path.basename(member.name)
            # Define the target file path in the output_dir
            target_path = os.path.join(output_dir, base_filename)

            # Write the file data to the new file
            with open(target_path, 'wb') as f:
                shutil.copyfileobj(file_data, f)

# Close the tar file
tar.close()


#creating a dataframe that has the patient id, file path, type

# Define the directory where the files are stored
directory = 'extracted_images_400x'

# Get the list of file names
filenames = os.listdir(directory)

# Create a DataFrame from the file names
df = pd.DataFrame(filenames, columns=['filename'])

# Extract the type (M or B), patient ID and image ID from the filename
df['type'] = df['filename'].str.split('_', expand=True)[1]
df['patient_ID'] = df['filename'].str.split('-', expand=True)[2]

# Display the DataFrame
df

# Get a list of unique patient IDs for each type
malignant_ids = df[df['type'] == 'M']['patient_ID'].unique()
benign_ids = df[df['type'] == 'B']['patient_ID'].unique()

# Function to split patient IDs into train, validation, and test
def split_ids(ids):
    np.random.shuffle(ids)
    train_size = int(len(ids) * 0.8)
    val_size = int(len(ids) * 0.1)
    train_ids = ids[:train_size]
    val_ids = ids[train_size:train_size + val_size]
    test_ids = ids[train_size + val_size:]
    return train_ids, val_ids, test_ids

# Split patient IDs for each type
train_m_ids, val_m_ids, test_m_ids = split_ids(malignant_ids)
train_b_ids, val_b_ids, test_b_ids = split_ids(benign_ids)

# Concatenate train, validation, and test patient IDs
train_patient_ids = np.concatenate([train_m_ids, train_b_ids])
val_patient_ids = np.concatenate([val_m_ids, val_b_ids])
test_patient_ids = np.concatenate([test_m_ids, test_b_ids])

# Get corresponding dataframes
train_df = df[df['patient_ID'].isin(train_patient_ids)]
val_df = df[df['patient_ID'].isin(val_patient_ids)]
test_df = df[df['patient_ID'].isin(test_patient_ids)]


print('file name is', 'train_df')
for column in train_df.columns:
    print(column, train_df[column].nunique())


print('file name is', 'val_df')
for column in val_df.columns:
    print(column, val_df[column].nunique())


print('file name is', 'test_df')
for column in test_df.columns:
    print(column, test_df[column].nunique())

# Function to plot a couple of sample images
def plot_sample_images(df, directory, class_name, num_samples=2):
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        filename = df[df['type'] == class_name]['filename'].iloc[i]
        img_path = os.path.join(directory, filename)
        img = plt.imread(img_path)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.show()


# Plot a couple of sample images from the train set
plot_sample_images(train_df, directory, 'M', num_samples=2)
plot_sample_images(train_df, directory, 'B', num_samples=2)

# Plot a couple of sample images from the validation set
plot_sample_images(val_df, directory, 'M', num_samples=2)
plot_sample_images(val_df, directory, 'B', num_samples=2)

# Plot a couple of sample images from the test set
plot_sample_images(test_df, directory, 'M', num_samples=2)
plot_sample_images(test_df, directory, 'B', num_samples=2)


# Define image size and path
img_size = (700, 460)
batch_size = 32
directory = 'extracted_images_400x'

# Rescale the pixel values of the images and rotate the images by an angle between -10 to +10 degrees
# Create an image data generator for training data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
# Create an image data generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=directory,
    x_col='filename',
    y_col='type',
    target_size=img_size,
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,  # Shuffle the training data
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=directory,
    x_col='filename',
    y_col='type',
    target_size=img_size,
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle the validation data
)
