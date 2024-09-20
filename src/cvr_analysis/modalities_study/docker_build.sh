VERSION=$1
NAME="cvr-analysis-modalities"

caffeinate -i docker buildx build \
	--platform linux/amd64,linux/arm64 \
	-t gusma78/$NAME:$VERSION \
	-t gusma78/$NAME:latest \
	--push \
	.