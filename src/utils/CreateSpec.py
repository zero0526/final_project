import random
import xml.etree.ElementTree as ET
import os
import json
from typing import Any, Dict, List
import logging
from xml.etree.ElementTree import Element
from pathlib import Path


logger = logging.getLogger(__name__)
class SNDLibLoad:
    def __init__(self, xml_path:str, config: Any):
        self.xml_path = xml_path
        self.config = config

    def load(self):
        if not os.path.exists(self.xml_path):
            raise FileExistsError(f"File xml dont exists please check the path= {self.xml_path}")

        try:
            tree= ET.parse(self.xml_path)
            root= tree.getroot()
            ns= {}
            if "}" in root.tag:
                ns_url= root.tag.split("}")[0].strip("{")
                ns= {"ns": ns_url}

            # 1. Parse Nodes & Assign Specs
            nodes_data = self.__parse_nodes(root, ns)

            # 2. Parse Links
            links_data = self.__parse_links(root, ns)

            data= {
                "nodes_data": nodes_data,
                "links_data": links_data,
                "topology": self.config.topology,
            }
            self.__save_json(data)
            return data
        except ET.ParseError as e:
            print(f"Syntax error XML: {e}")
        except Exception as e:
            print(f"Error : {e}")
            raise e

    def __parse_nodes(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        nodes= []
        xml_nodes: List[Element]= root.findall(".//ns:node", ns) if ns else root.findall(".//node")
        nodes_type= self.config.nodes_type[self.config.topology]
        logger.info(f"xml_nodes: {len(xml_nodes)}")
        for xml_node in xml_nodes:
            nid= xml_node.get("id")
            coords_elem= self.__find_node(xml_node, ns, "coordinates")
            x_tag= self.__find_node(coords_elem, ns, "x")
            y_tag= self.__find_node(coords_elem, ns, "y")
            node_type= nodes_type[nid] if nid in nodes_type else 'relay'
            node_obj={
                "id": nid,
                "type": node_type,
                "coordinates":{
                    "x": x_tag.text,
                    "y": y_tag.text,
                },
                "specs": self.__determine_resource(node_type)
            }
            nodes.append(node_obj)
        return nodes


    def __determine_resource(self, node_type:str)-> Dict[str, Any]:
        type_config = self.config.nodes_config[node_type]
        cpu= round(random.uniform(type_config.get("cpu_min"), type_config.get("cpu_max")), 2)
        return {
            "cpu": cpu,
            "ram": type_config.get("ram"),
            "hdd": type_config.get("hdd"),
            "is_computing": node_type!="relay"
        }

    def __parse_links(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, str]]:
        links = []
        xml_links = root.findall(".//ns:link", ns) if ns else root.findall(".//link")
        transmission_rate_min = self.config.transmission_rate.get("min")
        transmission_rate_max = self.config.transmission_rate.get("max")
        for link in xml_links:
            src_tag = self.__find_node(link, ns, "source")
            tgt_tag = self.__find_node(link, ns, "target")
            lid = link.get("id")

            links.append({
                "id": lid,
                "source": src_tag.text,
                "target": tgt_tag.text,
                "transmission_rate": round(random.uniform(transmission_rate_min, transmission_rate_max), 4)
            })
        return links

    def __find_node(self, root: ET.Element, ns: Dict[str, str], path:str, is_all: bool= False) -> Element | None | list[Element]:
        if is_all:
            return root.findall(f"ns:{path}", ns) if ns else root.findall(path, ns)
        return root.find(f"ns:{path}", ns) if ns else root.find(path, ns)

    def __save_json(self, data: Dict[str, Any]):
        # Determine the project root (3 levels up from src/utils/CreateSpec.py)
        project_root = Path(__file__).parent.parent.parent.resolve()
        save_path = project_root / "data" / f"{self.config.topology}_nodes_config.json"
        with open(save_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)



