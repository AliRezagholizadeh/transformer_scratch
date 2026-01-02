AMD_DIGITAL_OVEAN_PAYLOAD = {
  "name": "2.6.0---ROCm-7.0-gpu-mi300x1-192gb-devcloud-atl1",
  "region": "atl1",
  "size": "gpu-mi300x1-192gb-devcloud",
  "image": "amd-pytorchrocm7",
  "ssh_keys": [],
  "backups": False,
  "ipv6": False,
  "monitoring": False,
  "tags": [],
  "user_data": "",
  "vpc_uuid": ""
}
c_url = """curl -X POST -H 'Content-Type: application/json' \
    -H \"Authorization: Bearer '$TOKEN'\" \
    -d \"{'name': '2.6.0---ROCm-7.0-gpu-mi300x1-192gb-devcloud-atl1',
        'region': 'atl1',
        'size': 'gpu-mi300x1-192gb-devcloud',
        'image': 'amd-pytorchrocm7',
        'ssh_keys':[],
        'backups': False,
        'ipv6': False,
        'monitoring': False,
        'tags':[],
        'user_data': '',
        'vpc_uuid': ''}\" \
    https://api-amd.digitalocean.com/v2/droplets"""

class AMD_Digital_Ocean:
    def __init__(self, connection_parameter):
        self.connection_parameter = connection_parameter

    def connect(self):
        pass