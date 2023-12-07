from peaceful_pie.unity_comms import UnityComms

class Sampler:
    def __init__(self):
        pass

    def load_object(self):
        """
        Loads in the desired object from ShapeNet
    
        Args:
            object_id (str): The desired object
    
        Returns:
            The desired object
        """
        pass

    def train_model(self, object_type=None, additional_viewpoints=None):
        """
        Samples the desired object and produces a NeRF using Instant-NGP
    
        Args:
            object_type
            additional_viewpoints
    
        Returns:
            Images of the resulting NeRF
        """
        pass

    def produce_mesh(self, object_type=None, additional_viewpoints=None):
        """
        Samples the desired object and produces a NeRF using Instant-NGP
    
        Args:
            object_type
            additional_viewpoints
    
        Returns:
            Resulting mesh object file
        """
        pass

def run(msg):
    unity_comms = UnityComms(port=9000)
    unity_comms.Say(message=msg)

if __name__ == "__main__":
    run("Hello world")