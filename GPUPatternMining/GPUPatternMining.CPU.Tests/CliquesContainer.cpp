#include "catch.hpp"
#include "../GPUPatternMining.Contract/CliquesContainer.h"
#include "../GPUPatternMining.Contract/SubcliquesContainer.h"


TEST_CASE("CliqueAlreadyProcessed", "CliquesContainerTests")
{
	SubcliquesContainer container;

	std::vector<unsigned short> clique1 { 1, 2, 3, 4 };
	std::vector<unsigned short> clique2 { 0, 1, 3, 4 };
	std::vector<unsigned short> clique3 { 0, 1 };
	std::vector<unsigned short> clique4 { 5, 6, 7 };
	std::vector<unsigned short> clique5 { 0, 3, 4 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == true);
}

TEST_CASE("CliqueNotProcessed", "CliquesContainerFindingTests")
{
	SubcliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };
	std::vector<unsigned short> clique3{ 0, 1 };
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 0, 3, 5 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == false);
}

TEST_CASE("Smaller clique", "CliquesContainerFindingTests")
{
	SubcliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };
	std::vector<unsigned short> clique3{ 0, 1 };
	std::vector<unsigned short> clique4{ 5, 6, 7 };
	std::vector<unsigned short> clique5{ 1, 3, 4 };

	container.insertClique(clique1);
	container.insertClique(clique2);
	container.insertClique(clique3);
	container.insertClique(clique4);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique5);

	REQUIRE(cliqueAlreadyProcessed == true);
}

TEST_CASE("same size", "CliquesContainerFindingTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };

	std::vector<unsigned short> clique3{ 0, 1, 3, 4 };

	container.insertClique(clique1);
	container.insertClique(clique2);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique3);

	REQUIRE(cliqueAlreadyProcessed == true);
}

TEST_CASE("same size not", "CliquesContainerFindingTests")
{
	CliquesContainer container;

	std::vector<unsigned short> clique1{ 1, 2, 3, 4 };
	std::vector<unsigned short> clique2{ 0, 1, 3, 4 };

	std::vector<unsigned short> clique3{ 0, 1, 3, 5};

	container.insertClique(clique1);
	container.insertClique(clique2);

	bool cliqueAlreadyProcessed = container.checkCliqueExistence(clique3);

	REQUIRE(cliqueAlreadyProcessed == false);
}