package com.example.testingfiturecapstone

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.testingfiturecapstone.databinding.ActivityMainBinding
import com.example.testingfiturecapstone.ml.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.schema.TensorType.UINT8
import org.tensorflow.lite.schema.Uint8Vector
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var imageView: ImageView
    private lateinit var button: Button
    private lateinit var buttononigiri: Button
    private lateinit var buttonpredict: Button
    private lateinit var buttonfoodcategory: Button
    private lateinit var deasesbutton: Button
    private lateinit var tvOutput: TextView
//    private val GALLERY_REQUEST_CODE = 123
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        imageView = binding.imageView
        button = binding.btnCaptureImage
        tvOutput = binding.tvOutput
        deasesbutton = binding.deases
        val buttonLoad = binding.btnLoadImage
        button.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                takePicturePreview.launch(null)
            } else {
                requestPermission.launch(android.Manifest.permission.CAMERA)
            }
        }
        buttonLoad.setOnClickListener {
            startGallery()
        }
        buttononigiri = binding.predictOnigiri
        buttonpredict = binding.anotherPredict
        buttonfoodcategory = binding.anotherPredictFood
        buttononigiri.setOnClickListener {
            outputGeneratoronigiri(bitmap)
        }
        buttonpredict.setOnClickListener {
            outputGeneratormobile(bitmap)
        }
        buttonfoodcategory.setOnClickListener {
            outputGeneratorcategoryfood(bitmap)
        }
        deasesbutton.setOnClickListener {
            outputGeneratorcategorydeases(bitmap)
        }

    }

    private fun outputGeneratorcategorydeases(bitmapinput: Bitmap){
        val name_file = "labels.txt"
        val label = application.assets.open(name_file).bufferedReader().use { it.readText() }
        val labels = label.split("\n")


        val model = Model.newInstance(this)
        val bitmap = Bitmap.createScaledBitmap(bitmapinput, 150, 150, true)
        val input = ByteBuffer.allocateDirect(150*150*3*4).order(ByteOrder.nativeOrder())
        for (y in 0 until 150) {
            for (x in 0 until 150) {
                val px = bitmap.getPixel(x, y)

                // Get channel values from the pixel value.
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                val rf = (r - 127) / 255f
                val gf = (g - 127) / 255f
                val bf = (b - 127) / 255f

                input.putFloat(rf)
                input.putFloat(gf)
                input.putFloat(bf)
            }
        }
        imageView.setImageBitmap(bitmap)



        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 3), DataType.FLOAT32)
        val tensorImage = TensorImage(DataType.FLOAT32)
//        tensorImage.load()

//        val byteBuffer = tensorImage.buffer
        Log.d("shape", input.toString())
        Log.d("shape", inputFeature0.buffer.toString())
//        inputFeature0.loadBuffer(input)

//        // Runs model inference and gets result.
//        val outputs = model.process(inputFeature0)
//        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//
//        val max = getMax(outputFeature0.floatArray, outputFeature0.floatArray.size)
//        Log.e("outputGenerator: ", "-----------------------")
//        Log.e("outputGenerator: ", outputFeature0.floatArray.toList().toString())
//        Log.e("outputGenerator: ", max.toString())
//        Log.e("outputGenerator: ", outputFeature0.floatArray.size.toString())
//        Log.e("outputGenerator: ", outputFeature0.dataType.toString())
//        Log.e("outputGenerator: ", outputFeature0.dataType.toString())
//        tvOutput.text = labels[max]
        model.close()

    }



    private fun startGallery() {
        val intent = Intent()
        intent.action = Intent.ACTION_GET_CONTENT
        intent.type = "image/*"
        val chooser = Intent.createChooser(intent, "Choose a Picture")
        launcherIntentGallery.launch(chooser)
    }

    private val launcherIntentGallery = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val selectedImg: Uri = result.data?.data as Uri
            //data image
            bitmap =
                BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImg))
            bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)


        }
    }
    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                takePicturePreview.launch(null)
            } else {
                Toast.makeText(this, "Permission Denied !! Try again", Toast.LENGTH_SHORT).show()
            }
        }

    //launch camera and take picture
    private val takePicturePreview =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { it ->
            if (it != null) {

                bitmap = Bitmap.createScaledBitmap(it, 224, 224, true)


            }
        }

    private fun outputGeneratoronigiri(bitmap: Bitmap) {
        val name_file = "labels.txt"
        val label = application.assets.open(name_file).bufferedReader().use { it.readText() }
        val labels = label.split("\n")
        val model = Model1654305306.newInstance(this)
        var bitmapscale = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        imageView.setImageBitmap(bitmapscale)
        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmapscale)

        val byteBuffer = tensorImage.buffer
        Log.d("shape", byteBuffer.toString())
        Log.d("shape", inputFeature0.buffer.toString())
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer


        val max = getMax(outputFeature0.floatArray, outputFeature0.floatArray.size)
        Log.e("outputGenerator: ", "-----------------------")
        Log.e("outputGenerator: ", outputFeature0.floatArray.toList().toString())
        Log.e("outputGenerator: ", max.toString())
        Log.e("outputGenerator: ", outputFeature0.floatArray.size.toString())
        Log.e("outputGenerator: ", outputFeature0.dataType.toString())
        Log.e("outputGenerator: ", outputFeature0.dataType.toString())
        tvOutput.text = labels[max]
        model.close()
    }

    private fun outputGeneratormobile(bitmap: Bitmap) {
        val name_file = "label.txt"
        val label = application.assets.open(name_file).bufferedReader().use { it.readText() }
        val labels = label.split("\n")
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val model = MobilenetV110224Quant.newInstance(this)

        val tbuffer = TensorImage.fromBitmap(resized)
        val byteBuffer = tbuffer.buffer
        imageView.setImageBitmap(resized)
        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        Log.e("outputGenerator: ", "-----------------------")
        Log.e("outputGenerator: ", outputFeature0.floatArray.toList().toString())
        var max = getMax(outputFeature0.floatArray, 1000)

        tvOutput.text = labels[max]

        // Releases model resources if no longer used.
        model.close()
    }

    private fun outputGeneratorcategoryfood(bitmap: Bitmap) {
        //declearing tensor flow lite model variable

        val birdsModel = LiteModelAiyVisionClassifierFoodV11.newInstance(this)

        // converting bitmap into tensor flow image
        val newBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val tfimage = TensorImage.fromBitmap(newBitmap)

        //process the image using trained model and sort it in descending order
        val outputs = birdsModel.process(tfimage)
            .probabilityAsCategoryList.apply {
                sortByDescending { it.score }
            }
        imageView.setImageBitmap(newBitmap)
        //getting result having high probability
        val highProbabilityOutput = outputs[0]

        //setting ouput text
        tvOutput.text = highProbabilityOutput.label
        Log.i("TAG", "outputGenerator: $highProbabilityOutput")

    }

    fun getMax(arr: FloatArray, size: Int): Int {
        var ind = 0;
        var min = 0.0f;

        for (i in 0 until size) {
//            Log.e("get: ", i.toString())
//            Log.e("get i: ", arr[i].toString())
            if (arr[i] > min) {
                Log.e("getMax: ", i.toString())
                Log.e("getMax: ", arr[i].toString())

                min = arr[i]
                ind = i;
            }
        }
        return ind
    }
}

