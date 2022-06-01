package com.example.testingfiturecapstone

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import com.example.testingfiturecapstone.ml.OnigiriModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var imageView: ImageView
    private lateinit var button: Button
    private lateinit var tvOutput: TextView
    private val GALLERY_REQUEST_CODE = 123


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        imageView = binding.imageView
        button = binding.btnCaptureImage
        tvOutput = binding.tvOutput
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
            val bitmap =
                BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImg))
            imageView.setImageBitmap(bitmap)
            outputGenerator(bitmap)
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
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap)
                outputGenerator(bitmap)
            }
        }
    private fun outputGenerator(bitmap: Bitmap) {
        val name_file = "labels.txt"
        val label = application.assets.open(name_file).bufferedReader().use { it.readText() }
        val labels = label.split("\n")
        val model = OnigiriModel.newInstance(this)
        var bitmap = bitmap
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
// Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val byteBuffer = tensorImage.buffer
        Log.d("shape", byteBuffer.toString())
        Log.d("shape", inputFeature0.buffer.toString())
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        var max = getMax(outputFeature0.floatArray,outputFeature0.floatArray.size)

        Log.e("outputGenerator: ",max.toString() )
        Log.e("outputGenerator: ", outputFeature0.floatArray.size.toString())
        tvOutput.setText(labels[max])
        model.close()
    }
    fun getMax(arr: FloatArray, size: Int) : Int{
        var ind = 0;
        var min = 0.0f;

        for(i in 0 until size)
        {
            Log.e("get: ",i.toString() )
            Log.e("get i: ",arr[i].toString() )
            if(arr[i] > min)
            {
                Log.e("getMax: ",i.toString() )
                Log.e("getMax: ",arr[i].toString() )

                min = arr[i]
                ind = i;
            }
        }
        return ind
    }
}