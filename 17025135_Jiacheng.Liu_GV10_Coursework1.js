function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Ray Tracer';
	UI.titleShort = 'RayTracerSimple';
	UI.numFrames = 1000;
	UI.maxFPS = 24;
	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `RaytracingDemoFS - GL`,
		id: `RaytracingDemoFS`,
		initialValue: `precision highp float;

struct PointLight {
  vec3 position;
  vec3 color;
};

struct Material {
  vec3  diffuse;
  vec3  specular;
  float glossiness;
  // Expand the material struct with the additional necessary information
  float reflectionIndex;     //index of reflection
  float IOR;                 //index of refraction
};

struct Sphere {
  vec3 position;
  float radius;
  Material material;
};

struct Plane {
  vec3 normal;
  float d;
  Material material;
};

struct Cylinder {
  vec3 position;
  vec3 direction; 
  float radius;
  Material material;
};

const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 2;

struct Scene {
  vec3 ambient;
  PointLight[lightCount] lights;
  Sphere[sphereCount] spheres;
  Plane[planeCount] planes;
  Cylinder[cylinderCount] cylinders;
};

struct Ray {
  vec3 origin;
  vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
  bool hit;
  float t;
  vec3 position;
  vec3 normal;
  Material material;
};

HitInfo getEmptyHit() {
  return HitInfo(
    false,          //bool hit
    0.0,            //float t
    vec3(0.0),      //position
    vec3(0.0),      //normal
  	// Depending on the material definition extension you make, this constructor call might need to be extened as well
    Material(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0)    //default material
	);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
  // Make t1 the smaller t
  if(t2 < t1)  {
    float temp = t1;
    t1 = t2;
    t2 = temp;
  }
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
  return t > tMin && t < tMax;
}

// Get the smallest t in an interval
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {
 
  sortT(t0, t1);
 
  // As t0 is smaller, test this first
  if(isTInInterval(t0, tMin, tMax)) {
  	smallestTInInterval = t0;
    return true;
  }
 
  // If t0 was not in the interval, still t1 could be
  if(isTInInterval(t1, tMin, tMax)) {
  	smallestTInInterval = t1;
    return true;
  } 
 
  // None was
  return false;
}



HitInfo intersectSphere(const Ray ray, const Sphere sphere, const float tMin, const float tMax) {
             
  //ray form origin to sphere
    vec3 to_sphere = ray.origin - sphere.position;
 
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);
     
      	float smallestTInInterval;
      	if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
          return getEmptyHit();
        }
     
      	vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;     

      	vec3 normal =
          	length(ray.origin - sphere.position) < sphere.radius + 0.001?
          	-normalize(hitPosition - sphere.position) :
      		normalize(hitPosition - sphere.position);     

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	sphere.material);
    }
    return getEmptyHit();
}

// Plane intersection
// Expression for plane is ax + by + cz = d, with normal vector plane.normal=（a, b, c)
// Expression for ray is (ray.origin + t * ray.direction)
// Combine two expressions and solve parameter t, if t exists, then ray intersects plane
HitInfo intersectPlane(const Ray ray,const Plane plane, const float tMin, const float tMax) {
    // Put the code for plane intersection here 
    
    // Substitute (x, y, z) with (ray.origin + t * ray.direction), and solve
    // dot((ray.origin + t * ray.direction), plane.normal) = plane.d
    // t = (plane.d - dot(ray.origin, plane.normal))/ dot(ray.direction, plane.normal)
	float fac1 = dot( ray.origin, plane.normal );
	float fac2 = dot( ray.direction, plane.normal );
  
    // If fac2 = 0.0, it means ray's direction is perpendicular to plane's normal so ray won't intersect with the plane and t won't exist
    // Otherwise, t exists and ray intersects the plane
	if ( fac2 != 0.0 ){
  		float t = - ( fac1 + plane.d ) / fac2;
 		float smallestTInInterval;
 
        // If t in the interval, the hit position will be in the screen and could be calculated by putting t into ray expression
  		if ( isTInInterval ( t, tMin, tMax )){
    		vec3 hitPosition = ray.origin + t * ray.direction;
            
            // Return HitInfo
   			return HitInfo(
    			true,
      			t,
      			hitPosition,
      			plane.normal,
      			plane.material);
 		} 
 	}
    
    // If t does not exist, return empty HitInfo
  	return getEmptyHit();
}

float lengthSquared(vec3 x) {
  return dot(x, x);
}


// Cylinder Intersection
// Expression for cylinder of radius r oriented along a line (culinder.position + cylinder.direction * t) is ((q - cylinder.position - dot(cylinder.direction, q - cylinder.position) * cylinder.direction)^2) - r^2 = 0, where q is a point on the cylinder 
// Expression for ray is (ray.origin + t * ray.direction)
// Combine two expressions and solve parameter t, if t exists, then ray intersects cylinder
HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder, const float tMin, const float tMax) {
	// Put the code for cylinder intersection here
  
    // Substitute q = (ray.origin + t * ray.direction) and solve
    // (ray.origin - cylinder.position + ray.direction * t - dot(cylinder.direction, ray.origin - cylinder.position + ray.direction * t) * cylinder.position)^2 - cylidner.radius ^ 2 = 0
    // Reduces to at^2 + bt + c = 0
    // a = (ray.direction - dot(ray.direction, cylinder.direction) * cylinder.direction)^2
    // b = 2.0 * dot (a, ray.origin - cylinder.position - dot (ray.origin - cylinder.position, cylinder.direction) * cylinder.direction)
    // c =  (ray.origin - cylinder.position - dot (ray.origin - cylinder.position, cylinder.direction) * cylinder.direction)^2 - cylinder.radius
  	vec3 to_cylinder = ray.origin - cylinder.position;
  	vec3 fac1 = ray.direction - dot( ray.direction, cylinder.direction) * cylinder.direction;
  	vec3 fac2 = to_cylinder - dot( to_cylinder, cylinder.direction) * cylinder.direction;
  	
    float a = lengthSquared( fac1 );
    float b = 2.0 * dot( fac1, fac2);
  	float c = lengthSquared( fac2 )- cylinder.radius * cylinder.radius;
  
    
    // For D = b^2 - 4ac, if D > 0.0 then t has two solutions. If t = 0.0 then t has one solution and t does not have a solution if D < 0.0.
	float D = b * b - 4.0 * a * c;
  	if ( D > 0.0 ) {
        // Solve t
    	float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);
        
        // Get smaller t for nearer hit point
        float smallestTInInterval;
      	if ( getSmallestTInInterval ( t0, t1, tMin, tMax, smallestTInInterval )) {
      		vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction; 
      	  
            // Compute normal vector for cylinder
            float t = dot(hitPosition - cylinder.position, cylinder.direction)/lengthSquared(cylinder.direction);
      		vec3 normal = hitPosition - cylinder.position - cylinder.direction * t ;
      		normal = normalize( normal );
      
            // Return HitInfo
      		return HitInfo(
        	    true,
         	    smallestTInInterval,
          	    hitPosition,
          	    normal,
                cylinder.material);
      }
    }
  
    // If ray does not go through the cylinder, return the empty HitInfo
    return getEmptyHit();
}

//get nearer HitInfo
HitInfo getBetterHitInfo(const HitInfo oldHitInfo, const HitInfo newHitInfo) {
	if(newHitInfo.hit)
  		if(newHitInfo.t < oldHitInfo.t)  // No need to test for the interval, this has to be done per-primitive
          return newHitInfo;
  	return oldHitInfo;
}

//get HitInfo of nearset object that intersect with the ray
HitInfo intersectScene(const Scene scene, const Ray ray, const float tMin, const float tMax) {
  HitInfo bestHitInfo;
  bestHitInfo.t = tMax;
  bestHitInfo.hit = false;
  for (int i = 0; i < cylinderCount; ++i) {
    bestHitInfo = getBetterHitInfo(bestHitInfo, intersectCylinder(ray, scene.cylinders[i], tMin, tMax));
  }
  for (int i = 0; i < sphereCount; ++i) {
    bestHitInfo = getBetterHitInfo(bestHitInfo, intersectSphere(ray, scene.spheres[i], tMin, tMax));
  }
  for (int i = 0; i < planeCount; ++i) {
    bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPlane(ray, scene.planes[i], tMin, tMax));
  }
 
  return bestHitInfo;
}


// Function to complete shadow test
// Once we get the direction of the primary ray, we check every object in the scene to see if it intersect any of them.
// In some cases, primary ray will intersect more than one object. When it happens, we choose the object whose intersection point is the closest one to the origin of the ray (eye). 
// Then we shoot a shdow ray from the intersection point to the light. If this shadow ray does not intersect any object on the way to light, the hit point is illuminated. If this shadow intersects an object before reaching light, that object casts a shadow on it.
vec3 shadeFromLight(
  const Scene scene,
  const Ray ray,
  const HitInfo hit_info,
  const PointLight light)
{
  vec3 hitToLight = light.position - hit_info.position;
 
  vec3 lightDirection = normalize(hitToLight);
  vec3 viewDirection = normalize(hit_info.position - ray.origin);
  vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);
  float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
  float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);
  
  // Put your shadow test here
  // Declare shadow ray shoots from hit point towards to the point light 
  // check if it intersect with any objects
  Ray shadowRay;
  shadowRay.origin = hit_info.position;
  shadowRay.direction = lightDirection;
  float visibility = 1.0;
  
  // Here is a typical error that easy to make. 
  // The shadow ray go to infinity distance from the hit point, so shadow will be casted as well even though it interacts an object whose position is set above the point light.
  // Therefore, the maximum t for shadow ray should not be higher than light ray's length. 
  HitInfo lightHitInfo = intersectScene ( scene, shadowRay, 0.0001, length(hitToLight) );   
 
  // If shadow ray hit an object, there will be a shadow on current object and visibility will be set as 0.0, which casts a shadow on the current object.
  if( lightHitInfo.hit ) {
       visibility = 0.0;
       return visibility * light.color ;                                                                                    
  }
  
  // If shadow ray hit light directily, the current hit point will be illuminated and visiblity will be set as 1.0.
  visibility = 1.0;
     
  return 	visibility *
    		light.color * (
    		specular_term * hit_info.material.specular +
      		diffuse_term * hit_info.material.diffuse);
}

vec3 background(const Ray ray) {
  // A simple implicit sky that can be used for the background
  return vec3(0.2) + vec3(0.8, 0.6, 0.5) * max(0.0, ray.direction.y);
}

// It seems to be a WebGL issue that the third parameter needs to be inout instea dof const on Tobias' machine
vec3 shade(const Scene scene, const Ray ray, inout HitInfo hitInfo) {
 
  	if(!hitInfo.hit) {
  		return background(ray);
  	}
 
    vec3 shading = scene.ambient * hitInfo.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
        shading += shadeFromLight(scene, ray, hitInfo, scene.lights[i]);
    }
    return shading;
}


Ray getFragCoordRay(const vec2 frag_coord) {
  	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(800, 400);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance)); 
 
  	return Ray(origin, direction);
}


// In this function, Schlick's approximation was used to apply fresnel effect.
// Schlick's approximation is a formula for aprroximating the contribution of Fresnel factor in the specular reflection of light from a non-conducting interface between two medium.
// r_sita = r0 + (1 - r0)( 1 - cos_sita) ^ 5
// r0 = (ior1 - ior2) / (ior1 + ior2)
float fresnel(const vec3 viewDirection, const vec3 normal, const float ior1, const float ior2) {
  	// Put your code to compute the Fresnel effect here
    float cosine = -dot( viewDirection, normal );
  	float r0 = (ior1 - ior2)/(ior1 + ior2);
    r0 *= r0;
    float fac1 = 1.0 - cosine;
    float r_sita = r0 + (1.0 - r0)* pow( fac1, 5.0);
    return r_sita;
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
     
    Ray initialRay = getFragCoordRay(fragCoord); 
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.0001, 10000.0); 
  	vec3 result = shade(scene, initialRay, initialHitInfo);
	
  	Ray currentRay;
  	HitInfo currentHitInfo;
  	
  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;

    // The initial strength of the reflection
  	float reflectionWeight = 1.0;
  
    // The initial medium is air
    float currentIOR=1.0;
  	
  	const int maxReflectionStepCount = 2;
  	for(int i = 0; i < maxReflectionStepCount; i++) {
     
      if(!currentHitInfo.hit) break;
     
      // Update this with the correct values
      reflectionWeight *= fresnel(currentRay.direction, currentHitInfo.normal, currentIOR, currentHitInfo.material.IOR) * currentHitInfo.material.reflectionIndex;
      
      // Declare the reflection ray
      Ray nextRay;
      
	  // Put your code to compute the reflection ray here
	  nextRay.origin = currentHitInfo.position;
      nextRay.direction = currentRay.direction - 2.0 * dot(currentRay.direction,currentHitInfo.normal) * currentHitInfo.normal;
      
      // See the reflection ray as the incident ray in next loop 
      currentRay = nextRay;
      currentHitInfo = intersectScene(scene, currentRay, 0.0001, 10000.0);     
           
      result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }
 
  	// Compute the refraction
  	currentRay = initialRay; 
  	currentHitInfo = initialHitInfo;
  
  	// The initial strength of the refraction.
  	float refractionWeight = 1.0;
 
  	const int maxRefractionStepCount = 2;
  	for(int i = 0; i < maxRefractionStepCount; i++) {
     
      if(!currentHitInfo.hit) break;

      // Update this with the correct values
      // Assume the the sum of reflection weight and refraction weight equals to 1
      refractionWeight *= 1.0 - fresnel( currentRay.direction, currentHitInfo.normal, currentIOR, currentHitInfo.material.IOR);
      
      // Declare the refraction ray
      Ray nextRay;
	  // Put your code to compute the reflection ray
      float IOR_ratio=currentIOR/currentHitInfo.material.IOR;
      // float fac1=dot(currentHitInfo.normal,currentRay.direction);
      // float fac2=sqrt(1.0+IOR_ratio*IOR_ratio*(fac1*fac1-1.0));
      nextRay.origin = currentHitInfo.position;
      nextRay.direction = refract ( 1.0 * currentRay.direction, currentHitInfo.normal, IOR_ratio);
      //nextRay.direction=-IOR_ratio*(-currentRay.direction)+(IOR_ratio*fac1-fac2)*currentHitInfo.normal;
      
      // See refraction ray as incident ray in next loop
      currentRay=nextRay;

      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
      currentIOR = currentHitInfo.material.IOR;
           
      result += refractionWeight * shade(scene, currentRay, currentHitInfo);
    }
  return result;
}

Material getDefaultMaterial() {
  // Will need to update this to match the new Material definition
  return Material(vec3(0.3), vec3(0), 1.0, 0.0, 0.0);
}

// White paper material
Material getPaperMaterial() {
  // Replace by your definition of a paper material
  return Material(vec3(1.0), vec3(0.0), 1.0, 0.0, 0.0);
}

// Yello plastic material
Material getPlasticMaterial() {
  // Replace by your definition of a plastic material
  return Material(vec3(250.0/255.0,250.0/255.0,70.0/255.0), vec3(1.0), 7.0, 0.0, 0.0);
}

Material getGlassMaterial() {
  // Replace by your definition of a glass material
  return Material(vec3(0.0), vec3(0.0), 0.5, 1.0, 1.12);
}

Material getSteelMirrorMaterial() {
  // Replace by your definition of a steel mirror material
  return Material(vec3(0.0),vec3(0.0),0.1,0.2,0.0);
}

vec3 tonemap(const vec3 radiance) {
  const float monitorGamma = 2.0;
  return pow(radiance, vec3(1.0 / monitorGamma));
}

void main()
{
    // Setup scene
    Scene scene;
  	scene.ambient = vec3(0.12, 0.15, 0.2);
 
    // Lights
    scene.lights[0].position = vec3(5, 15, -5);
    scene.lights[0].color    = 0.5 * vec3(0.8, 0.6, 0.5);
   
  	scene.lights[1].position = vec3(-15, 10, 2);
    scene.lights[1].color    = 0.5 * vec3(0.5, 0.7, 1.0);
 
    // Primitives
    scene.spheres[0].position            	= vec3(8, -2, -13);
    scene.spheres[0].radius              	= 4.0;
    scene.spheres[0].material 				= getPaperMaterial();
   
  	scene.spheres[1].position            	= vec3(-7, -1, -13);
    scene.spheres[1].radius             	= 4.0;
    scene.spheres[1].material				= getPlasticMaterial();
 
    scene.spheres[2].position            	= vec3(0, 0.5, -5);
    scene.spheres[2].radius              	= 2.0;
    scene.spheres[2].material   			= getGlassMaterial();

  	scene.planes[0].normal            		= vec3(0, 1, 0);
  	scene.planes[0].d              			= 4.5;
    scene.planes[0].material				= getSteelMirrorMaterial();
 
  	scene.cylinders[0].position            	= vec3(-1, 1, -18);
  	scene.cylinders[0].direction            = normalize(vec3(-1, 2, -1));
  	scene.cylinders[0].radius         		= 1.5;
    scene.cylinders[0].material				= getPaperMaterial();
 
  	scene.cylinders[1].position            	= vec3(3, 1, -5);
  	scene.cylinders[1].direction            = normalize(vec3(1, 4, 1));
  	scene.cylinders[1].radius         		= 0.25;
    scene.cylinders[1].material				= getPlasticMaterial();

  // compute color for fragment
  gl_FragColor.rgb = tonemap(colorForFragment(scene, gl_FragCoord.xy));
  gl_FragColor.a = 1.0;
}

`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RaytracingDemoVS - GL`,
		id: `RaytracingDemoVS`,
		initialValue: `attribute vec3 position;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
	try {
		gl = canvas.getContext("experimental-webgl");
		gl.viewportWidth = canvas.width;
		gl.viewportHeight = canvas.height;
	} catch (e) {
	}
	if (!gl) {
		alert("Could not initialise WebGL, sorry :-(");
	}
}

function getShader(gl, id) {
	var shaderScript = document.getElementById(id);
	if (!shaderScript) {
		return null;
	}

	var str = "";
	var k = shaderScript.firstChild;
	while (k) {
		if (k.nodeType == 3) {
			str += k.textContent;
		}
		k = k.nextSibling;
	}

	var shader;
	if (shaderScript.type == "x-shader/x-fragment") {
		shader = gl.createShader(gl.FRAGMENT_SHADER);
	} else if (shaderScript.type == "x-shader/x-vertex") {
		shader = gl.createShader(gl.VERTEX_SHADER);
	} else {
		return null;
	}

    console.log(str);
	gl.shaderSource(shader, str);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		alert(gl.getShaderInfoLog(shader));
		return null;
	}

	return shader;
}

function RaytracingDemo() {
}

RaytracingDemo.prototype.initShaders = function() {

	this.shaderProgram = gl.createProgram();

	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoVS"));
	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoFS"));
	gl.linkProgram(this.shaderProgram);

	if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
		alert("Could not initialise shaders");
	}

	gl.useProgram(this.shaderProgram);

	this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
	gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

	this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
	this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RaytracingDemo.prototype.initBuffers = function() {
	this.triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	
	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	this.triangleVertexPositionBuffer.itemSize = 3;
	this.triangleVertexPositionBuffer.numItems = 3 * 2;
}

RaytracingDemo.prototype.drawScene = function() {
			
	var perspectiveMatrix = new J3DIMatrix4();	
	perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

	var modelViewMatrix = new J3DIMatrix4();	
	modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);
		
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
	
	gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RaytracingDemo.prototype.run = function() {
	this.initShaders();
	this.initBuffers();

	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT);

	this.drawScene();
};

function init() {	
	

	env = new RaytracingDemo();	
	env.run();

    return env;
}

function compute(canvas)
{
    env.initShaders();
    env.initBuffers();

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    env.drawScene();
}