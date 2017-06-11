// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"
#include "../../ORUtils/Image.h"

#include <sstream>
#include <stdlib.h>

namespace ITMLib
{
	namespace Objects
	{
		class ITMMesh
		{
		public:
          	/// \brief Represents a 3D triangle with per-vertex color information.
			struct Triangle {
				Vector3f p0, p1, p2;
              	Vector3f c0, c1, c2;
			};
		
			MemoryDeviceType memoryType;

			uint noTotalTriangles;
			const uint noMaxTriangles;

			ORUtils::MemoryBlock<Triangle> *triangles;

			explicit ITMMesh(MemoryDeviceType memoryType, long sdfLocalBlockNum)
				: memoryType(memoryType),
				  noTotalTriangles(0),
				  noMaxTriangles(sdfLocalBlockNum * 32)
			{
				printf("Allocating memory block for mesh triangles. noMaxTriangles=%d.\n", noMaxTriangles);
				triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, memoryType);
			}

			void WriteOBJ(const char *fileName)
			{
				// TODO(andrei): The obj format doesn't support color officially. The best
				// alternative is to use the ply format, which is easy to parse AND supports
				// per-vertex color information officially.
				ORUtils::MemoryBlock<Triangle> *cpu_triangles;
				bool shouldDelete = false;
				if (memoryType == MEMORYDEVICE_CUDA)
				{
					cpu_triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
					cpu_triangles->SetFrom(triangles, ORUtils::MemoryBlock<Triangle>::CUDA_TO_CPU);
					shouldDelete = true;
				}
				else cpu_triangles = triangles;

				Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);

				FILE *f = fopen(fileName, "w+");
				if (noTotalTriangles > noMaxTriangles) {
					std::stringstream ss;
					ss << "Unable to save mesh to file [" << fileName << "]. Too many triangles: "
					   << noTotalTriangles << " while the maximum is " << noMaxTriangles << ".";
					throw std::runtime_error(ss.str());
				}

				if (f != NULL)
				{
					for (uint i = 0; i < noTotalTriangles; i++) {
						// TODO(andrei): Suport color dumping in STL models as well, if possible (may
						// be unsupported officialy by the format...).

						const Vector3f &c0 = triangleArray[i].c0;
						const Vector3f &c1 = triangleArray[i].c1;
						const Vector3f &c2 = triangleArray[i].c2;
						fprintf(f,
								"v %f %f %f %f %f %f\n",
								triangleArray[i].p0.x,
								triangleArray[i].p0.y,
								triangleArray[i].p0.z,
								c0.r,
								c0.g,
								c0.b);
						fprintf(f,
								"v %f %f %f %f %f %f\n",
								triangleArray[i].p1.x,
								triangleArray[i].p1.y,
								triangleArray[i].p1.z,
								c1.r,
								c1.g,
								c1.b);
						fprintf(f,
								"v %f %f %f %f %f %f\n",
								triangleArray[i].p2.x,
								triangleArray[i].p2.y,
								triangleArray[i].p2.z,
								c2.r,
								c2.g,
								c2.b);
					}

					for (uint i = 0; i<noTotalTriangles; i++) fprintf(f, "f %d %d %d\n", i * 3 + 2 + 1, i * 3 + 1 + 1, i * 3 + 0 + 1);
					fclose(f);
				}

				if (shouldDelete) delete cpu_triangles;
			}

			void WriteSTL(const char *fileName)
			{
				ORUtils::MemoryBlock<Triangle> *cpu_triangles; bool shoulDelete = false;
				if (memoryType == MEMORYDEVICE_CUDA)
				{
					cpu_triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
					cpu_triangles->SetFrom(triangles, ORUtils::MemoryBlock<Triangle>::CUDA_TO_CPU);
					shoulDelete = true;
				}
				else cpu_triangles = triangles;

				Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);

				FILE *f = fopen(fileName, "wb+");

				if (f != NULL) {
					for (int i = 0; i < 80; i++) fwrite(" ", sizeof(char), 1, f);

					fwrite(&noTotalTriangles, sizeof(int), 1, f);

					float zero = 0.0f; short attribute = 0;
					for (uint i = 0; i < noTotalTriangles; i++)
					{
						fwrite(&zero, sizeof(float), 1, f); fwrite(&zero, sizeof(float), 1, f); fwrite(&zero, sizeof(float), 1, f);

						fwrite(&triangleArray[i].p2.x, sizeof(float), 1, f); 
						fwrite(&triangleArray[i].p2.y, sizeof(float), 1, f); 
						fwrite(&triangleArray[i].p2.z, sizeof(float), 1, f);

						fwrite(&triangleArray[i].p1.x, sizeof(float), 1, f); 
						fwrite(&triangleArray[i].p1.y, sizeof(float), 1, f); 
						fwrite(&triangleArray[i].p1.z, sizeof(float), 1, f);

						fwrite(&triangleArray[i].p0.x, sizeof(float), 1, f);
						fwrite(&triangleArray[i].p0.y, sizeof(float), 1, f);
						fwrite(&triangleArray[i].p0.z, sizeof(float), 1, f);

						fwrite(&attribute, sizeof(short), 1, f);

						//fprintf(f, "v %f %f %f\n", triangleArray[i].p0.x, triangleArray[i].p0.y, triangleArray[i].p0.z);
						//fprintf(f, "v %f %f %f\n", triangleArray[i].p1.x, triangleArray[i].p1.y, triangleArray[i].p1.z);
						//fprintf(f, "v %f %f %f\n", triangleArray[i].p2.x, triangleArray[i].p2.y, triangleArray[i].p2.z);
					}

					//for (uint i = 0; i<noTotalTriangles; i++) fprintf(f, "f %d %d %d\n", i * 3 + 2 + 1, i * 3 + 1 + 1, i * 3 + 0 + 1);
					fclose(f);
				}

				if (shoulDelete) delete cpu_triangles;
			}

			~ITMMesh()
			{
				delete triangles;
			}

			// Suppress the default copy constructor and assignment operator
			ITMMesh(const ITMMesh&);
			ITMMesh& operator=(const ITMMesh&);
		};
	}
}
