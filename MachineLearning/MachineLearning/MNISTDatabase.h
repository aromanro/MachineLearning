#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "DataFileBase.h"

namespace Utils {

	class MNISTDatabase
	{
	public:
		MNISTDatabase() 
		{
			setRelativePath("../../Datasets/");
			setImagesFileName("emnist-digits-train-images-idx3-ubyte");
			setLabelsFileName("emnist-digits-train-labels-idx1-ubyte");
		}
		
		void setRelativePath(const std::string& p)
		{
			imagesFile.setRelativePath(p);
			labelsFile.setRelativePath(p);
		}
		
		void setImagesFileName(const std::string& d)
		{
			imagesFile.setDataFileName(d);
		}
		
		void setLabelsFileName(const std::string& d)
		{
			labelsFile.setDataFileName(d);
		}
		
		std::string getImagesFilePath() const
		{
			return imagesFile.getFilePath();
		}
		
		std::string getLabelsFilePath() const
		{
			return labelsFile.getFilePath();
		}

		bool Open()
		{
			imagesStream.open(imagesFile.getFilePath(), std::ios::binary);
			if (!imagesStream.good()) return false;
			// check magic
			uint32_t magic = 0;
			imagesStream.read((char*)&magic, sizeof(uint32_t));
			magic = ntohlAlt(magic);
			if (magic != 0x803) 
			{
				imagesStream.close();
				return false;
			}

			uint32_t cntImgs = 0;
			imagesStream.read((char*)&cntImgs, sizeof(uint32_t));
			cntImgs = ntohlAlt(cntImgs);

			uint32_t rows = 0;	
			imagesStream.read((char*)&rows, sizeof(uint32_t));
			rows = ntohlAlt(rows);

			uint32_t cols = 0;
			imagesStream.read((char*)&cols, sizeof(uint32_t));
			cols = ntohlAlt(cols);

			if (rows != cols || rows != imgSize)
			{
				imagesStream.close();
				return false;
			}

			labelsStream.open(labelsFile.getFilePath(), std::ios::binary);
			if (!labelsStream.good()) 
			{
				imagesStream.close();
				return false;
			}
			// check magic
			magic = 0;
			labelsStream.read((char*) &magic, sizeof(uint32_t));
			magic = ntohlAlt(magic);
			if (magic != 0x801) 
			{
				Close();

				return false;
			}

			uint32_t cntLabels = 0;
			labelsStream.read((char*)&cntLabels, sizeof(uint32_t));
			cntLabels = ntohlAlt(cntLabels);

			if (cntLabels != cntImgs)
			{
				Close();

				return false;
			}

			return true;
		}

		void Close()
		{
			imagesStream.close();
			labelsStream.close();
		}

		bool ReadImage(std::vector<double>& img)
		{
			if (!imagesStream.good()) return false;
			
			std::vector<uint8_t> imgbuf(imgSize * imgSize);
			imagesStream.read((char*)imgbuf.data(), imgbuf.size());
			
			img.resize(imgbuf.size());
			for (size_t i = 0; i < imgbuf.size(); ++i)
				img[i] = (double)imgbuf[i] / 255.;

			return true;
		}

		bool ReadLabel(uint8_t& label)
		{
			if (!labelsStream.good()) return false;
			labelsStream.read((char*)&label, sizeof(uint8_t));
			return true;
		}

		std::pair<std::vector<double>, uint8_t> ReadImageAndLabel()
		{
			std::vector<double> img;
			uint8_t label = 0xFF; // invalid, read values will be 0-9

			if (ReadImage(img))
				ReadLabel(label);
			
			return std::make_pair(img, label);
		}

		std::vector<std::pair<std::vector<double>, uint8_t>> ReadAllImagesAndLabels(bool augment = false)
		{
			std::vector<std::pair<std::vector<double>, uint8_t>> res;

			while (imagesStream.good() && labelsStream.good())
			{
				auto imgAndLabel = ReadImageAndLabel();
				if (imgAndLabel.second != 0xFF) 
				{
					if (augment)
					{
						std::vector<double> shiftedUp(imgAndLabel.first.begin() + imgSize, imgAndLabel.first.end());
						shiftedUp.resize(imgSize * imgSize, 0);
						res.emplace_back(make_pair(shiftedUp, imgAndLabel.second));

						std::vector<double> shiftedDown(imgSize * imgSize, 0);
						std::copy(imgAndLabel.first.begin(), imgAndLabel.first.begin() + shiftedDown.size() - imgSize, shiftedDown.begin() + imgSize);
						res.emplace_back(make_pair(shiftedDown, imgAndLabel.second));

						std::vector<double> shiftedLeft(imgSize * imgSize, 0);
						for (int i = 0; i < imgSize; ++i)
							std::copy(imgAndLabel.first.begin() + i * imgSize + 1, imgAndLabel.first.begin() + (i + 1) * imgSize, shiftedLeft.begin() + i * imgSize);
						res.emplace_back(make_pair(shiftedLeft, imgAndLabel.second));

						std::vector<double> shiftedRight(imgSize * imgSize, 0);
						for (int i = 0; i < imgSize; ++i)
							std::copy(imgAndLabel.first.begin() + i * imgSize, imgAndLabel.first.begin() + (i + 1) * imgSize - 1, shiftedRight.begin() + i * imgSize + 1);

						res.emplace_back(make_pair(shiftedRight, imgAndLabel.second));
					}

					res.emplace_back(imgAndLabel);
				}
				else break;
			}

			return res;
		}

	private:
		static uint32_t ntohlAlt(uint32_t val)
		{
			uint8_t buf[4];
			memcpy(buf, &val, 4);

			return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) | ((uint32_t)buf[2] << 8) | (uint32_t)buf[3];
		}

		DataFileBase imagesFile;
		DataFileBase labelsFile;

		std::ifstream imagesStream;
		std::ifstream labelsStream;

		const int imgSize = 28;
	};

}


