using System.Collections;
using System.Collections.Generic;
using System;
using System.Runtime.Serialization;
using UnityEngine;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

using AustinHarris.JsonRpc;

public class CameraViewpoint
{
    public float x;
    public float y;
    public float z;
    public float theta;
    public float phi;
    public CameraViewpoint(Vector3 v, Quaternion q) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        this.theta = 0;
        this.phi = 0;
    }
    public Vector3 GetPosition()
    {
        return new Vector3(this.x, this.y, this.z);
    }
    public Quaternion GetQuaternion() 
    {
        Quaternion rotation = Quaternion.Euler(-90 + this.phi, -this.theta, 0f);
        return rotation;
    }
}

[RequireComponent(typeof(Camera))]
public class CameraRotator : MonoBehaviour
{

    class Rpc : JsonRpcService {
        
        Camera rpc_camera;
        GameObject rpc_target_object;
        List<Vector3> rpc_positions_list;
        List<Quaternion> rpc_quaternions_list;
        Vector3 sc = new Vector3();
        private string positions_fp = "Assets/screenshots/positions.txt";
        private string quaternions_fp = "Assets/screenshots/quaternions.txt";
        private int curr_poses = 0;
        
        public Rpc(Camera rpc_camera, GameObject rpc_target_object, List<Vector3> rpc_positions_list, List<Quaternion> rpc_quaternions_list) {
            this.rpc_camera = rpc_camera;
            this.rpc_target_object = rpc_target_object;
            this.rpc_positions_list = rpc_positions_list;
            this.rpc_quaternions_list = rpc_quaternions_list;
        }

        [JsonRpcMethod]
        void sample_default_viewpoints(int max_poses) {
            // Move the camera around in a singular ring and capture 50 poses
            rpc_camera.transform.position = new Vector3(0.0f, 5.0f, 10.0f);
            rpc_camera.transform.LookAt(rpc_target_object.transform.position);
            sc = getSphericalCoordinates(rpc_camera.transform.position);

            while (curr_poses < max_poses) {
                // Update camera position
                float dx = (2 * Mathf.PI) / max_poses;
                sc.y += dx;
                rpc_camera.transform.position = getCartesianCoordinates(sc) + rpc_target_object.transform.position;
                rpc_camera.transform.LookAt(rpc_target_object.transform.position);
                rpc_positions_list.Add(rpc_camera.transform.position);
                rpc_quaternions_list.Add(rpc_camera.transform.rotation);

                // Take screenshot
                string filename = "Assets/screenshots/pic" + curr_poses.ToString() + ".png";
                RenderTexture active_rt = RenderTexture.active;
                RenderTexture.active = rpc_camera.targetTexture; 
                rpc_camera.Render();
                Texture2D image = new Texture2D(rpc_camera.targetTexture.width, rpc_camera.targetTexture.height);
                image.ReadPixels(new Rect(0, 0, rpc_camera.targetTexture.width, rpc_camera.targetTexture.height), 0, 0);
                image.Apply();
                RenderTexture.active = active_rt;
                byte[] bytes = image.EncodeToPNG();
                Destroy(image);
                File.WriteAllBytes(filename, bytes);

                // Update pose count
                curr_poses += 1;
            }
        }


        [JsonRpcMethod]
        void sample_additional_viewpoints(CameraViewpoint pose) {
            rpc_camera.transform.position = pose.GetPosition();
            rpc_camera.transform.rotation = pose.GetQuaternion();
            // Write position/quaternion info
            rpc_positions_list.Add(rpc_camera.transform.position);
            rpc_quaternions_list.Add(rpc_camera.transform.rotation);

            // Capture camera pose
            string filename = "Assets/screenshots/pic" + curr_poses.ToString() + ".png";
            RenderTexture active_rt = RenderTexture.active;
            RenderTexture.active = rpc_camera.targetTexture; 
            rpc_camera.Render();
            Texture2D image = new Texture2D(rpc_camera.targetTexture.width, rpc_camera.targetTexture.height);
            image.ReadPixels(new Rect(0, 0, rpc_camera.targetTexture.width, rpc_camera.targetTexture.height), 0, 0);
            image.Apply();
            RenderTexture.active = active_rt;
            byte[] bytes = image.EncodeToPNG();
            Destroy(image);
            File.WriteAllBytes(filename, bytes);

            // Update pose count
            curr_poses += 1;

        }

        [JsonRpcMethod]
        // After sampling all desired poses, write position/quaternion info to files
        void sampling_cleanup() {
            write_positions(positions_fp, rpc_positions_list);
            write_quaternions(quaternions_fp, rpc_quaternions_list);
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
    static public void write_positions(string file_path, List<Vector3> matrices)
    {
        StreamWriter writer = new StreamWriter(file_path, true);
        for (int i = 0; i < matrices.Count; i++) {
            writer.WriteLine(matrices[i].ToString());
        }
        writer.Close();
    }

    // Write the list of camera quarternions to a file
    static public void write_quaternions(string file_path, List<Quaternion> matrices)
    {
        StreamWriter writer = new StreamWriter(file_path, true);
        for (int i = 0; i < matrices.Count; i++) {
            writer.WriteLine(matrices[i].ToString());
        }
        writer.Close();
    }

    // Convert Cartesian coordinates to Spherical coordinates
    static public Vector3 getSphericalCoordinates(Vector3 cartesian) {
        float r = Mathf.Sqrt(Mathf.Pow(cartesian.x, 2) + Mathf.Pow(cartesian.y, 2) + Mathf.Pow(cartesian.z, 2));
        float phi = Mathf.Atan2(cartesian.y, cartesian.x);
        float theta = Mathf.Acos(cartesian.z / r);
        return new Vector3(r, phi, theta); 
    }

    // Convert Spherical coordinates to Cartesian coordinates
    static public Vector3 getCartesianCoordinates(Vector3 spherical) {
        Vector3 result = new Vector3();
        result.x = spherical.x * Mathf.Cos(spherical.z) * Mathf.Cos(spherical.y);
        result.y = spherical.x * Mathf.Sin(spherical.z);
        result.z = spherical.x * Mathf.Cos(spherical.z) * Mathf.Sin(spherical.y);
        return result;
    }

    // Create a new targetTexture before the script runs
    void Awake() {
        Camera = GetComponent<Camera>();
        Camera.gameObject.SetActive(true);
        Camera.targetTexture = new RenderTexture(image_width, image_height, 24);
    }

    void Start()
    {

        // Delete all the existing camera poses in the screenshots directory
        string directoryPath = "./Assets/screenshots/";

        // Check if the directory exists
        if (Directory.Exists(directoryPath))
        {
            Debug.Log("Directory contents are being emptied...");
            // Delete all files within the directory
            foreach (string filePath in Directory.GetFiles(directoryPath))
            {
                File.Delete(filePath);
            }
            Debug.Log("Directory contents emptied successfully.");
        }
        else
        {
            Debug.Log("Directory does not exist.");
        }
        
        start_time = Time.time;

        num_screenshots = 0;

        positions_list = new List<Vector3>();
        quaternions_list = new List<Quaternion>();
        positions_file_path = "Assets/screenshots/positions.txt";
        quaternions_file_path = "Assets/screenshots/quaternions.txt";

        rpc = new Rpc(Camera, target_object, positions_list, quaternions_list);

        // Get all the parameters of the camera, and write it to a file
        camera_properties = new List<float>(); 
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

}