using System.Collections;
using System.Collections.Generic;
using System;
using System.Runtime.Serialization;
using UnityEngine;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

using AustinHarris.JsonRpc;

[RequireComponent(typeof(Camera))]
public class CameraRotator : MonoBehaviour
{

    class Rpc : JsonRpcService {
        [JsonRpcMethod]
        void Say(string message) {
            Debug.Log(message);
        }
    }

    Rpc rpc;

    public GameObject target_object;
    private Camera Camera;
    private Transform center_point;
    public float radius = 5f;

    private int image_width = 850;
    private int image_height = 531;

    public int max_screenshots;
    private int num_screenshots;

    private List<Vector3> positions_list;
    private string positions_file_path;
    private Vector3 curr_position;

    private List<Quaternion> quaternions_list;
    private string quaternions_file_path;
    private Quaternion curr_quaternion;

    private List<float> camera_properties;

    private float start_time;
    private float elapsed_time;

    Vector3 target = new Vector3(0, 0, 0);
    Vector3 sc = new Vector3();

    // Write the list of camera positions to a file
    public void write_positions(string file_path, List<Vector3> matrices)
    {
        StreamWriter writer = new StreamWriter(file_path, true);
        for (int i = 0; i < matrices.Count; i++) {
            writer.WriteLine(matrices[i].ToString());
        }
        writer.Close();
    }

    // Write the list of camera quarternions to a file
    public void write_quaternions(string file_path, List<Quaternion> matrices)
    {
        StreamWriter writer = new StreamWriter(file_path, true);
        for (int i = 0; i < matrices.Count; i++) {
            writer.WriteLine(matrices[i].ToString());
        }
        writer.Close();
    }

    // Function to get the intrinsic properties of the camera
    private Matrix4x4 GetIntrinsic(Camera cam)
    {
        float pixel_aspect_ratio = (float)cam.pixelWidth / (float)cam.pixelHeight;

        float alpha_u = cam.focalLength * ((float)cam.pixelWidth / cam.sensorSize.x);
        float alpha_v = cam.focalLength * pixel_aspect_ratio * ((float)cam.pixelHeight / cam.sensorSize.y);

        float u_0 = (float)cam.pixelWidth / 2;
        float v_0 = (float)cam.pixelHeight / 2;

        //IntrinsicMatrix in row major
        Matrix4x4 camIntriMatrix = new Matrix4x4(new Vector4(alpha_u, 0f, u_0, 0f),
                                                 new Vector4(0f, alpha_v, v_0, 0f),
                                                 new Vector4(0f, 0f, 1f, 0f),
                                                 new Vector4(0f, 0f, 0f, 0f));
        return camIntriMatrix;
    }

    // Convert Cartesian coordinates to Spherical coordinates
    private Vector3 getSphericalCoordinates(Vector3 cartesian) {
        float r = Mathf.Sqrt(Mathf.Pow(cartesian.x, 2) + Mathf.Pow(cartesian.y, 2) + Mathf.Pow(cartesian.z, 2));
        float phi = Mathf.Atan2(cartesian.y, cartesian.x);
        float theta = Mathf.Acos(cartesian.z / r);
        return new Vector3(r, phi, theta); 
    }

    // Convert Spherical coordinates to Cartesian coordinates
    private Vector3 getCartesianCoordinates(Vector3 spherical) {
        Vector3 result = new Vector3();
        result.x = spherical.x * Mathf.Cos(spherical.z) * Mathf.Cos(spherical.y);
        result.y = spherical.x * Mathf.Sin(spherical.z);
        result.z = spherical.x * Mathf.Cos(spherical.z) * Mathf.Sin(spherical.y);
        return result;
    }

    // After each iteration of Update(), capture image with Unity camera
    void LateUpdate() {
        Capture();
    }


    // Captures an image using the Unity camera
    public void Capture() {
        string filename = "Assets/screenshots/pic" + (num_screenshots - 1).ToString() + ".png";

        RenderTexture active_rt = RenderTexture.active;
        RenderTexture.active = Camera.targetTexture; 

        Camera.Render();

        // Current issue: Camera's targetTexture is Null value
        Texture2D image = new Texture2D(Camera.targetTexture.width, Camera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, Camera.targetTexture.width, Camera.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = active_rt;

        byte[] bytes = image.EncodeToPNG();
        Destroy(image);

        File.WriteAllBytes(filename, bytes);

    }

    // Create a new targetTexture before the script runs
    void Awake() {
        Camera = GetComponent<Camera>();
        Camera.gameObject.SetActive(true);
        Camera.targetTexture = new RenderTexture(image_width, image_height, 24);
    }

    void Start()
    {
        rpc = new Rpc();

        start_time = Time.time;

        num_screenshots = 0;
        Application.targetFrameRate = 300;

        positions_list = new List<Vector3>();
        quaternions_list = new List<Quaternion>();
        positions_file_path = "Assets/screenshots/positions.txt";
        quaternions_file_path = "Assets/screenshots/quaternions.txt";

        // Get all the parameters of the camera, and write it to a file
        camera_properties = new List<float>(); 
        Matrix4x4 intrinsic = GetIntrinsic(Camera);
        float fl_x = 567.0f;
        float fl_y = 567.0f;
        float cx = 425.0f;
        float cy = 268.0f;
        float camera_angle_x = 1.29f;
        float camera_angle_y = 0.88f;
        // We can assume all of the Unity cameras have no distortion, so we can set the distortion parameters to 0
        float k1 = 0f;
        float k2 = 0f;
        float p1 = 0f;
        float p2 = 0f;
        camera_properties.Add(camera_angle_x); // camera_angle_x
        camera_properties.Add(camera_angle_y); // camera_angle_y
        camera_properties.Add(fl_x); // fl_x
        camera_properties.Add(fl_y); // fl_y
        camera_properties.Add(k1); // k1
        camera_properties.Add(k2); // k2
        camera_properties.Add(p1); // p1
        camera_properties.Add(p2); // p2
        camera_properties.Add(cx); // cx
        camera_properties.Add(cy); // cy
        string camera_properties_file_path = "Assets/screenshots/camera_properties.txt";
        StreamWriter camera_properties_writer = new StreamWriter(camera_properties_file_path, true);
        for (int i = 0; i < camera_properties.Count; i++) {
            camera_properties_writer.WriteLine(camera_properties[i].ToString());
        }
        camera_properties_writer.Close();

        target = target_object.transform.position;

    }
    
    void Update() {
        elapsed_time = Time.time - start_time;
        int images_per_level = 50;
        string file_path = "Assets/screenshots/pic" + num_screenshots.ToString() + ".png";
        if (num_screenshots >= 0 && num_screenshots <= 49) {
            // TODO: determine dx based on max_screenshots
            if (num_screenshots == 0) {
                this.transform.position = new Vector3(0.0f, 2.0f, 10.0f);
                this.transform.LookAt(target);
                sc = getSphericalCoordinates(this.transform.position);
            }

            // calculate dx based on max_screenshots
            float dx = (2 * Mathf.PI) / images_per_level;
            // Update phi to rotate around the y-axis
            sc.y += dx;
            transform.position = getCartesianCoordinates(sc) + target;
            transform.LookAt(target);
            curr_position = Camera.main.transform.position;
            curr_quaternion = Camera.main.transform.rotation;
            positions_list.Add(curr_position);
            quaternions_list.Add(curr_quaternion);
        } else if (num_screenshots >= 50 && num_screenshots <= 99) {
            // TODO: determine dx based on max_screenshots
            if (num_screenshots == 50) {
                this.transform.position = new Vector3(0.0f, 5.0f, 10.0f);
                this.transform.LookAt(target);
                sc = getSphericalCoordinates(this.transform.position);
            }
            
            // calculate dx based on max_screenshots
            float dx = (2 * Mathf.PI) / images_per_level;
            // Update phi to rotate around the y-axis
            sc.y += dx;
            transform.position = getCartesianCoordinates(sc) + target;
            transform.LookAt(target);
            curr_position = Camera.main.transform.position;
            curr_quaternion = Camera.main.transform.rotation;
            positions_list.Add(curr_position);
            quaternions_list.Add(curr_quaternion);
        }
        num_screenshots++;
        if (num_screenshots >= 100) {
            write_positions(positions_file_path, positions_list);
            write_quaternions(quaternions_file_path, quaternions_list);
            UnityEditor.EditorApplication.isPlaying = false;
            Debug.Log("Elapsed Time: " + elapsed_time);
        }

    }

}