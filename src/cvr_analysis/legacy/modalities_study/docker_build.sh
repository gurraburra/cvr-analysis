NAME="cvr-analysis-modalities"
VERSION=$(python -c "from cvr_analysis._version import __version__; print(__version__.replace('+','_'))")

caffeinate -i docker buildx build \
	--platform linux/amd64,linux/arm64 \
	-t gusma78/$NAME:$VERSION \
	-t gusma78/$NAME:latest \
	--push \
	.