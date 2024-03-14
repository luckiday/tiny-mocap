import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r122/build/three.module.js';
import {
    OrbitControls
} from 'https://threejsfundamentals.org/threejs/resources/threejs/r122/examples/jsm/controls/OrbitControls.js';
import {
    GLTFLoader
} from 'https://threejsfundamentals.org/threejs/resources/threejs/r122/examples/jsm/loaders/GLTFLoader.js';

let avatarAnimations = JSON.parse(animationData);


async function updateMotion(landmarks) {
    let url = 'capture_motion';
    let data = {'id': 0, 'time': (new Date()).getTime(), 'ldmk': landmarks};

    let res = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    if (res.ok) {
        let text = await res.text();
        JSON.parse(text)
        return text;
        // let ret = await res.json();
        // return ret.data;
    } else {
        return `HTTP error: ${res.status}`;
    }
}

function afterIkTransformation(response) {
    // Assign the animation to the avatar
    response = JSON.parse(response);
    // if (debuggingMode) {
    //     document.getElementById("frame_parameters").innerHTML = response["frame"];
    // }
    transDict = response;
}

function avatar_animation() {
    const canvas = document.getElementsByClassName('avatar_canvas')[0];
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
    renderer.shadowMap.enabled = true;
    renderer.setClearColor(0x000000, 0); // the default

    //renderer.autoClear = true; //USELESS

    const fov = 25;
    const aspect = 320 / 180;  // the canvas default
    const near = 0.1;
    const far = 200;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.set(0, 20, 80);

    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 15, 0);
    controls.update();
    controls.enabled = false;


    const scene = new THREE.Scene();

    // scene.background = new THREE.Color('#779c6f');
    // const loader = new THREE.TextureLoader();
    // loader.load('web/bg-server.jpg', function (texture) {
    //     scene.background = texture;
    // });

    let avatarBones = {};
    {
        const skyColor = 0xFFFFFF;
        const groundColor = 0x888888;
        const intensity = 8;
        const light = new THREE.HemisphereLight(skyColor, groundColor, intensity);
        scene.add(light);
    }

    {
        const color = 0xFFFFFF;
        const intensity = 1;
        const light = new THREE.DirectionalLight(color, intensity);
        light.castShadow = true;
        light.position.set(0, 0, 0);
        light.target.position.set(-1, -1, -1);

        light.shadow.bias = -0.004;
        light.shadow.mapSize.width = 1;
        light.shadow.mapSize.height = 1;

        scene.add(light);
        scene.add(light.target);
        const cam = light.shadow.camera;
        cam.near = 1;
        cam.far = 10;
        cam.left = -10;
        cam.right = 10;
        cam.top = 10;
        cam.bottom = -10;

        const cameraHelper = new THREE.CameraHelper(cam);
        scene.add(cameraHelper);
        cameraHelper.visible = false;
        const helper = new THREE.DirectionalLightHelper(light, 100);
        scene.add(helper);
        helper.visible = false;

        function updateCamera() {
            // update the light target's matrixWorld because it's needed by the helper
            light.updateMatrixWorld();
            light.target.updateMatrixWorld();
            helper.update();
            // update the light's shadow camera's projection matrix
            light.shadow.camera.updateProjectionMatrix();
            // and now update the camera helper we're using to show the light's shadow camera
            cameraHelper.update();
        }

        updateCamera();
    }

    function resizeRendererToDisplaySize(renderer) {
        const canvas = renderer.domElement;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const needResize = canvas.width !== width || canvas.height !== height;
        if (needResize) {
            renderer.setSize(width, height, false);
        }
        return needResize;
    }


    function dumpObject(obj, lines = [], isLast = true, prefix = '') {
        const localPrefix = isLast ? '└─' : '├─';
        lines.push(`${prefix}${prefix ? localPrefix : ''}${obj.name || '*no-name*'} [${obj.type}]`);
        const newPrefix = prefix + (isLast ? '  ' : '│ ');
        const lastNdx = obj.children.length - 1;
        obj.children.forEach((child, ndx) => {
            const isLast = ndx === lastNdx;
            dumpObject(child, lines, isLast, newPrefix);
        });
        return lines;
    }

    function frameArea(sizeToFitOnScreen, boxSize, boxCenter, camera) {
        const halfSizeToFitOnScreen = sizeToFitOnScreen;
        const halfFovY = THREE.MathUtils.degToRad(camera.fov);
        const distance = halfSizeToFitOnScreen / Math.tan(halfFovY);
        // compute a unit vector that points in the direction the camera is now
        // in the xz plane from the center of the box
        const direction = (new THREE.Vector3())
            .subVectors(camera.position, boxCenter)
            // .multiply(new THREE.Vector3(1, 0, 1))
            .normalize();

        // move the camera to a position distance units way from the center
        // in whatever direction the camera was from the center already
        camera.position.copy(direction.multiplyScalar(distance).add(boxCenter));
        // pick some near and far values for the frustum that
        // will contain the box.
        camera.near = boxSize / 100;
        camera.far = boxSize * 100;
        camera.updateProjectionMatrix();

        // point the camera to look at the center of the box
        camera.lookAt(boxCenter.x, boxCenter.y, boxCenter.z);
    }

    function addPart(part) {
        // console.log(part);
        const partChildren = part.children.slice();
        const len = partChildren.length;
        for (let i = 0; i < len; ++i) {
            // console.log(partChildren[i].type)
            // console.log(partChildren[i])
            if (partChildren[i].type === "Bone") {
                avatarBones[partChildren[i].name] = partChildren[i];
            }
        }
        partChildren.forEach((child, i) => {
            addPart(child);
        });
    }


    {
        let loadedPart;
        const gltfLoader = new GLTFLoader();
        gltfLoader.load('web/avatar_v3.glb', (gltf) => {
            // console.log(dumpObject(gltf.scene).join('\n'));
            const root = gltf.scene;
            scene.add(root);
            root.traverse((obj) => {
                if (obj.castShadow !== undefined) {
                    obj.castShadow = true;
                    obj.receiveShadow = true;
                }
            });
            loadedPart = root.getObjectByName('Armature1');
            root.updateMatrixWorld();
            addPart(loadedPart);
            // console.log(avatarBones);
            const box = new THREE.Box3().setFromObject(root);
            const boxSize = box.getSize(new THREE.Vector3()).length();
            const boxCenter = box.getCenter(new THREE.Vector3());
            frameArea(boxSize * 1, boxSize, boxCenter, camera);
            // controls.maxDistance = boxSize * 10;
            // controls.target.copy(boxCenter);
            // controls.update();
        });
    }

    let counter = 0;

    function checkAnimationUpdate() {
        counter++;
        if (counter % 50 === 1) {
            document.getElementById("bg-image").src = "http://127.0.0.1:8000/background/" + counter.toString() + ".jpg";
        }
    }

    // setInterval(checkAnimationUpdate, 3000);

    function render(time) {
        checkAnimationUpdate();
        // Update background texture every 5 seconds
        // time *= 0.001;
        updateMotion("").then(afterIkTransformation);
        const total = Object.keys(avatarAnimations).length;

        // updateFrameNumber(frame);
        if (resizeRendererToDisplaySize(renderer)) {
            const canvas = renderer.domElement;
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        }

        let frameNumber = Math.floor(time * 0.02) % (total - 1) + 1;
        if (transDict != null) {
            console.log(transDict)
            for (const [key, value] of Object.entries(transDict)) {
                try {
                    avatarBones["Body1" + key].quaternion.set(...value);
                } catch (error) {
                    console.error(error);
                }
            }
        } else {
            for (const [key, value] of Object.entries(avatarAnimations[frameNumber])) {
                try {
                    avatarBones["Body1" + key].quaternion.set(...value);
                } catch (error) {
                    console.error(error);
                }
            }
        }
        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
}

avatar_animation()